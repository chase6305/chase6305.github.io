#!/usr/bin/env python3
"""Inspect PEAR's pinned protocol, recorded methods and data-manifest products.

No model, third-party package, simulator, device, network or package initializer.
The source revision and file hashes are verified BEFORE any source is executed.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

REVISION = "2a81b2eec95e6e6f19c3fc0a80a38aebb40a30e2"
HASHES = {
    "pear/agent_loop/protocol.py": "511e71155d44d925945189a00c871a6c9c4d3a1c193bbae11e0f8ce0f13ba99a",
    "pear/agent_loop/tool_parser.py": "3ce4e2fa60b3291a69c99c3321c11eaaa0b671ce5ab95c54a67a69f4785cf8bd",
    "pear/envs/robodojo/controller.py": "b9d30e4ccb31bfd986bb837cb1b4dec40899b8b0b65d05247f7068fab49ed763",
    "pear/deploy/station/agent_loop/action_parser.py": "6a46b351751cb36255f9f21c5566ad2356042dff757ed4b1b7d66cb15647a386",
    "pear/agent_loop/dataset.py": "eee246ad5cbfb54c85a3b6628348ad1f2f928b9db6b7ba0b7978a6bbf1272ee1",
    "pear/envs/libero/configs/libero_all.json": "ed2cc87605655eeafcf456e0a193a4a9569f226fe050bd7cf75efb9b52d2d352",
    "pear/envs/robodojo/configs/pickup_align_all.json": "efe64048b87977313e4792a30e74c4e1244fb80e00c4d08b3062f3a34db87ce4",
    "pear/envs/robodojo/configs/official_smoke.json": "09e720cb7a69c0a95d232ebbe91f12e8d4d7da6311fa9424e2bd20fa9ac5a8fd",
    "examples/models/qwen3_8_27b_pear/train/run_stage3.sh": "36b3f0204702c2bfed525fc1dbeabb1f07ed3a260316efcc87598b8ddb213c96"
}

def verified_source(source):
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git","-C",str(source),"rev-parse","HEAD"],text=True,stderr=subprocess.PIPE,timeout=10
    ).strip()
    if revision != REVISION:
        raise ValueError("Source commit mismatch: " + revision)
    texts = {}
    for relative, expected in HASHES.items():
        data = (source/relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Source hash mismatch: " + relative)
        texts[relative] = data.decode("utf-8")
    return source, texts

def extracted(text, names, namespace, filename):
    tree = ast.parse(text, filename=filename)
    selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise ValueError("Expected source class absent")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")],level=0), *selected],
        type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module),filename,"exec"),namespace)
    return namespace

def inspect_manifests(source, texts):
    """Expand pinned row products with the source's pure range method only.

    No datasets/verl import, environment constructor or rollout is performed.
    Repeated files change nominal row weights, rather than unique initial states.
    """
    dataset_name="pear/agent_loop/dataset.py"
    tree=ast.parse(texts[dataset_name],filename=str(source/dataset_name))
    dataset=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="EpisodeDataset")
    method=next(n for n in dataset.body if isinstance(n,ast.FunctionDef) and n.name=="_parse_range")
    method.decorator_list=[]
    fragment=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),method],type_ignores=[])
    namespace={}
    exec(compile(ast.fix_missing_locations(fragment),str(source/dataset_name),"exec"),namespace)
    parse_range=namespace["_parse_range"]
    assert parse_range("0-3,7")==[0,1,2,3,7]
    assert parse_range("0-28,30-54")==list(range(29))+list(range(30,55))
    def row_counts(name):
        manifest=json.loads(texts[name]);counts=Counter();unique=set()
        products=manifest["env"] if isinstance(manifest["env"],list) else [manifest["env"]]
        for product in products:
            keys=list(product)
            values=[parse_range(value) if isinstance(value,str) else list(value) for value in product.values()]
            for combination in itertools.product(*values):
                row=dict(zip(keys,combination))
                counts[str(row[manifest["data_source"]])]+=1
                unique.add(tuple(sorted(row.items())))
        return {"rows":sum(counts.values()),"unique_rows":len(unique),"by_data_source":dict(sorted(counts.items()))}
    libero=row_counts("pear/envs/libero/configs/libero_all.json")
    dojo=row_counts("pear/envs/robodojo/configs/pickup_align_all.json")
    smoke=row_counts("pear/envs/robodojo/configs/official_smoke.json")
    assert libero["rows"]==libero["unique_rows"]==2000
    assert set(libero["by_data_source"].values())=={500} and len(libero["by_data_source"])==4
    assert dojo["rows"]==dojo["unique_rows"]==328
    assert dojo["by_data_source"]=={"align_blocks":165,"general_pickup":163}
    assert smoke["rows"]==smoke["unique_rows"]==len(smoke["by_data_source"])==52
    assert not ({"align_blocks","general_pickup"}&set(smoke["by_data_source"]))
    recipe_name="examples/models/qwen3_8_27b_pear/train/run_stage3.sh"
    match=re.search(r"^PICKUP_ALIGN_REPEAT_N=(\d+)\s*$",texts[recipe_name],re.MULTILINE)
    assert match is not None
    repeat=int(match.group(1));assert repeat==6
    return {"libero_all":libero,"pickup_align_all":dojo,"official_smoke":smoke,
            "stage3_robodojo_file_repetitions":repeat,
            "stage3_nominal_row_weights":{"libero":libero["rows"],"robodojo":dojo["rows"]*repeat},
            "scope":"Pinned task/initial-state row products and literal recipe weight; not generated rollouts, scheduler frequencies, retained batches or a full benchmark run."}

def expect_rejection(operation):
    try:
        operation()
    except (ValueError, OverflowError) as error:
        return type(error).__name__
    raise AssertionError("Invalid action accepted")

def run(source, texts):
    sys.dont_write_bytecode = True
    protocol_name = "pear/agent_loop/protocol.py"
    protocol_space = {"__name__":"verified_pear_protocol"}
    exec(compile(texts[protocol_name],str(source/protocol_name),"exec"),protocol_space)
    protocol = protocol_space["JsonBlockProtocol"]
    schema = {"arm":("x","y","z","roll","pitch","yaw"),"gripper":("width",)}
    valid = {"arm":dict.fromkeys(schema["arm"],0),"gripper":{"width":0}}
    protocol.check_action(valid,schema)
    invalid = [None,[],{},dict(valid,unknown={})]
    missing = deepcopy(valid); del missing["arm"]["x"]; invalid.append(missing)
    extra = deepcopy(valid); extra["arm"]["unknown"]=0; invalid.append(extra)
    for bad in (True,"0",None,float("nan"),float("inf"),-float("inf"),10**400):
        payload = deepcopy(valid); payload["arm"]["x"]=bad; invalid.append(payload)
    protocol_rejections=[]
    for payload in invalid:
        protocol_rejections.append(expect_rejection(lambda payload=payload: protocol.check_action(payload,schema)))

    # Real deployment parser, with only its final model validation replaced by
    # a recorder. This does not check Pydantic or the robot action models.
    class Recorder:
        calls = 0
        @classmethod
        def model_validate(cls, value):
            cls.calls += 1
            return deepcopy(value)
    deployment_name = "pear/deploy/station/agent_loop/action_parser.py"
    deployment_space = extracted(texts[deployment_name],["ActionParser"],
        {"json":json,"math":math,"JsonBlockProtocol":protocol},str(source/deployment_name))
    deployment = deployment_space["ActionParser"]
    format_ = SimpleNamespace(schema=schema,action_model=Recorder)
    fence = "```json\n"+json.dumps(valid)+"\n```"
    assert deployment.parse(fence,format_) == valid
    deployment_rejections=[]
    for payload in invalid:
        deployment_rejections.append(expect_rejection(lambda payload=payload:deployment.parse(
            "```json\n"+json.dumps(payload)+"\n```",format_)))
    duplicate = fence.replace('"x": 0','"x": 1, "x": 0',1)
    for answer in (fence+"\ntrailing text","<think>private</think>\n"+fence,duplicate,"no action"):
        expect_rejection(lambda answer=answer:deployment.parse(answer,format_))

    # Training extraction is a separate path. Fake tokenizer and call wrapper
    # replace only their interfaces; no generation or verl package is loaded.
    parser_name = "pear/agent_loop/tool_parser.py"
    training_space = extracted(texts[parser_name],["JsonBlockToolParser","NoToolCall"],
        {"json":json,"JsonBlockProtocol":protocol,"ToolParser":type("ToolParser",(),{}),
         "FunctionCall":lambda **kw:SimpleNamespace(**kw),"get_event_loop":asyncio.get_running_loop},
        str(source/parser_name))
    async def training_checks():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        parser = training_space["JsonBlockToolParser"]()
        parser.tokenizer = SimpleNamespace(decode=lambda _ids:parser.response)
        tools = [SimpleNamespace(function=SimpleNamespace(name="record_robot_action"))]
        parser.response = fence+"\ntrailing text"
        _, calls = await parser.parse([],tools)
        assert len(calls)==1 and json.loads(calls[0].arguments)==valid
        parser.response = "```json\n{broken}\n```\n"+fence
        _, calls = await parser.parse([],tools)
        assert len(calls)==1 and json.loads(calls[0].arguments)==valid
        parser.response = fence+"\n```json\n{broken}\n```"
        _, calls = await parser.extract_tool_calls([],tools)
        assert calls == []
        parser.response = "no action"
        _, calls = await parser.extract_tool_calls([],tools)
        assert calls == []
        return {"last_json_block":True,"invalid_last_block_does_not_fall_back":True,
                "trailing_text_extracted":True,"missing_action_rejected":True}
    training = asyncio.run(training_checks())

    # Extract ONLY _goals. Its dependencies are state and coordinate recorders,
    # never a simulator or the full controller class.
    controller_name = "pear/envs/robodojo/controller.py"
    tree=ast.parse(texts[controller_name],filename=str(source/controller_name))
    controller = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="MotionController")
    method=next(n for n in controller.body if isinstance(n,ast.FunctionDef) and n.name=="_goals")
    fragment=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),method],type_ignores=[])
    goal_space={}
    exec(compile(ast.fix_missing_locations(fragment),str(source/controller_name),"exec"),goal_space)
    quantities = {
        "left_arm":{"x":20.045,"y":15.03,"z":35.11,"roll":.11,"pitch":.22,"yaw":.33},
        "right_arm":{"x":20.045,"y":-15.03,"z":35.11,"roll":.11,"pitch":.22,"yaw":.33},
    }
    recorded=[]
    def pose_record(values):
        recorded.append(deepcopy(values))
        return tuple(values[name] for name in ("x","y","z","roll","pitch","yaw"))
    controller_stub=type("RecordedController",(),{
        "ARMS":("left","right"),"_goals":goal_space["_goals"],
        "_measured_quantities":lambda _self:deepcopy(quantities),
        "tool_pose_of_quantities":staticmethod(pose_record),
    })()
    action={part:dict.fromkeys(schema["arm"],0) for part in ("left_arm","right_arm")}
    action.update(left_gripper={"width":0},right_gripper={"width":0})
    altered=deepcopy(action); altered["left_arm"]["z"]=-5; altered["left_arm"]["yaw"]=.1
    goals=controller_stub._goals(altered)
    assert set(goals)=={"left"} and math.isclose(goals["left"]["pose"][2],30.11)
    assert math.isclose(goals["left"]["pose"][5],.43)
    assert quantities["left_arm"]["z"]==35.11
    hold=controller_stub._goals(action)
    assert set(hold)=={"left","right"} and all(g["opening"] is None for g in hold.values())
    assert math.isclose(hold["left"]["pose"][0],20.045)
    for change,expected in ((1,1.0),(100,1.0),(-1,0.0),(-100,0.0)):
        altered=deepcopy(action); altered["left_gripper"]["width"]=change
        goals=controller_stub._goals(altered)
        assert goals=={"left":{"pose":None,"opening":expected}}
    return {"source_commit":REVISION,"verified_files":list(HASHES),
            "data_manifests":inspect_manifests(source,texts),
            "protocol_invalid_rejections":len(invalid),
            "protocol_exception_types":sorted(set(protocol_rejections)),
            "deployment_exception_types":sorted(set(deployment_rejections)),
            "deployment_requires_final_fence":True,
            "deployment_rejects_duplicate_fields":True,"training_extraction":training,
            "recorded_target_z_cm":30.11,"unrounded_current_quantity_preserved":True,
            "zero_action_records_holds":True,"gripper_sign_not_magnitude":True,
            "scope":"Pure protocol, extracted methods and manifest products only; no full action-model validation, dataset framework, IK, simulation, policy or hardware."}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.exit(2,"Output already exists; preserved: "+str(args.output)+"\n")
    try:
        source,texts=verified_source(args.source)
        report=run(source,texts)
        encoded=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
        if args.output:
            with args.output.open("x") as sink:
                sink.write(encoded)
        print(encoded,end="")
    except (ValueError,OSError,subprocess.SubprocessError) as error:
        parser.exit(2,str(error)+"\n")

if __name__=="__main__":
    main()
