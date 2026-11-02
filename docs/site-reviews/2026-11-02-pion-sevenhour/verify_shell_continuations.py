"""Parse reformatted commands with Bash and a fake Python; no probe is run."""
from pathlib import Path
import os,json,subprocess,tempfile,argparse

def commands(path):
 lines=path.read_text().splitlines();result=[];i=0
 while i<len(lines):
  line=lines[i].lstrip()
  if line.startswith('python -B '):
   parts=[line]
   while parts[-1].endswith('\\'):
    i+=1;parts.append(lines[i].lstrip())
   command='\n'.join(parts)
   if '--official-file' in command:result.append(command)
  i+=1
 return result

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--before-article',type=Path,required=True);parser.add_argument('--before-readme',type=Path,required=True);parser.add_argument('--article',type=Path,required=True);parser.add_argument('--readme',type=Path,required=True);parser.add_argument('--python',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();rows=[]
 with tempfile.TemporaryDirectory(prefix='pion argv ')as name:
  folder=Path(name);bin_dir=folder/'bin';bin_dir.mkdir();log=folder/'argv.jsonl';fake=bin_dir/'python'
  fake.write_text('#!'+args.python+' -B\nimport json,os,sys\nwith open(os.environ["PION_ARGV_CAPTURE"],"a") as log:log.write(json.dumps(sys.argv[1:])+"\\n")\n');fake.chmod(0o755)
  env={**os.environ,'PATH':str(bin_dir)+os.pathsep+os.environ['PATH'],'PION_ARGV_CAPTURE':str(log)}
  for label,before,after in [('article',args.before_article,args.article),('readme',args.before_readme,args.readme)]:
   old,new=commands(before),commands(after);assert len(old)==len(new)>0
   for index,(a,b)in enumerate(zip(old,new),1):
    values=[]
    for command in(a,b):
     log.write_text('');checked=subprocess.run(['bash','-e','-c',command],cwd=folder,env=env,capture_output=True,text=True,timeout=10);assert checked.returncode==0,checked.stderr
     actual=[json.loads(line)for line in log.read_text().splitlines()];assert len(actual)==1,actual;values.append(actual[0])
    assert values[0]==values[1],(a,b,values)
    rows.append({'source':label,'command_index':index,'before':a,'after':b,'argv':values[0],'exactly_one_command':True,'argv_equal':True})
 record={'scope':__doc__,'cases':rows,'pairs':len(rows),'bash_executions':2*len(rows),'probe_executions':0,'installations':0};args.output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'pairs':len(rows),'bash_executions':2*len(rows),'argv_equal':True,'probe_executions':0}))
if __name__=='__main__':main()
