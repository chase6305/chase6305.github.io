/* Pion's normalized scalar maps: no training data, network or GPU dependency. */
(function () {
  "use strict";
  const promotion = Object.freeze([1.875, -1.25, .375]);
  const suppression = Object.freeze([0, 2.5, -1.5]);
  const muonCoefficients = Object.freeze([3.4445, -4.7750, 2.0315]);
  const presets = Object.freeze({
    concentrated: Object.freeze({raw: [.8, .4, .2, .02], kp: 1, epsilon: false}),
    flat: Object.freeze({raw: [1, 1, 1, 1], kp: 0, epsilon: false}),
    tiny: Object.freeze({raw: [8e-9, 4e-9, 2e-9, 2e-10], kp: 1, epsilon: true}),
    zero: Object.freeze({raw: [0, 0, 0, 0], kp: 1, epsilon: false})
  });
  function nonnegative(value) {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) throw new RangeError("Use finite nonnegative amplitudes");
    return value;
  }
  function checkSteps(kp) {
    if (!Number.isInteger(kp) || kp < 0 || kp > 5) throw new RangeError("Use an integer promotion count from 0 to 5");
    return kp;
  }
  function polynomial(value, coefficients) {
    const square = value*value;
    return value*(coefficients[0] + square*(coefficients[1] + coefficients[2]*square));
  }
  function response(sigma, kp, muon = false) {
    nonnegative(sigma); checkSteps(kp);
    if (typeof muon !== "boolean") throw new TypeError("The Muon choice must be boolean");
    if (sigma > 1) throw new RangeError("Use a normalized singular value in [0, 1]");
    for (let index = 0; index < 5; index++) sigma = polynomial(sigma, muon ? muonCoefficients : index < kp ? promotion : suppression);
    return sigma;
  }
  function amplitudes(values) {
    if (!Array.isArray(values) || values.length < 1 || values.length > 64) throw new RangeError("Use one to 64 amplitudes");
    return Array.from(values,nonnegative);
  }
  function normalize(values, epsilon = 0) {
    const raw = amplitudes(values); nonnegative(epsilon);
    const maximum = Math.max(...raw);
    if (maximum === 0) return raw.map(() => 0);
    const scaled = raw.map(value => value/maximum), norm = Math.hypot(...scaled);
    if (epsilon > maximum) {
      const ratio = maximum/epsilon;
      return scaled.map(value => (value*ratio)/(1+norm*ratio));
    }
    return scaled.map(value => value/(norm+epsilon/maximum));
  }
  function effectiveRank(values) {
    const raw = amplitudes(values), maximum = Math.max(...raw);
    if (maximum === 0) return 0;
    const scaled = raw.map(value => value/maximum), total = scaled.reduce((sum,value) => sum+value, 0);
    return Math.exp(-scaled.reduce((sum,value) => {
      const probability = value/total;
      return probability > 0 ? sum+probability*Math.log(probability) : sum;
    }, 0));
  }
  function calculate(values, kp = 1, epsilon = 0) {
    const raw = amplitudes(values); checkSteps(kp); nonnegative(epsilon);
    const normalized = normalize(raw,epsilon), muon = normalized.map(value => response(value,0,true));
    const pion = normalized.map(value => response(value,kp));
    return {raw, normalized, muon, pion, input_erank: effectiveRank(raw),
      muon_erank: effectiveRank(muon), pion_erank: effectiveRank(pion),
      muon_norm: Math.hypot(...muon), pion_norm: Math.hypot(...pion),
      promotion_steps: kp, suppression_steps: 5-kp, epsilon, all_zero: raw.every(value => value === 0)};
  }
  function format(value) {
    if (value === 0) return "0";
    return Math.abs(value) < 1e-4 ? value.toExponential(5) : Number(value.toPrecision(6)).toString();
  }
  function writeNumber(node,value) {
    if(value===null){node.textContent="—";return;}
    const text=format(value),exponent=text.indexOf("e");
    // A narrow result card may wrap before the exponent, never between its
    // sign and digits. The selectable text remains the exact numeric string.
    if(exponent<0)node.textContent=text;
    else {
      const tail=document.createElement("span");tail.className="pion-spectrum__exponent";tail.textContent=text.slice(exponent);
      node.replaceChildren(text.slice(0,exponent),document.createElement("wbr"),tail);
    }
  }
  const svgNS = "http://www.w3.org/2000/svg";
  function draw(svg,result,requestedWidth) {
    const width = Math.max(180,requestedWidth ?? svg.getBoundingClientRect().width), font = parseFloat(getComputedStyle(svg).fontSize);
    const height = Math.max(230,font*14), yTicks = [0,.5,1,1.25];
    const measure = document.createElementNS(svgNS,"text");
    measure.setAttribute("visibility","hidden"); svg.append(measure);
    const labelWidth = Math.max(...yTicks.map(value => {measure.textContent=String(value);return measure.getBBox().width;}));
    measure.remove();
    const left=Math.ceil(Math.max(48,labelWidth+15)),right=16,top=18,bottom=Math.max(38,font*2.6);
    const plotWidth=Math.max(40,width-left-right),plotHeight=height-top-bottom;
    const x=value => left+value*plotWidth,y=value => top+(1.25-value)/1.25*plotHeight,nodes=[];
    function add(tag,attributes,text) {
      const node=document.createElementNS(svgNS,tag);
      for(const [key,value] of Object.entries(attributes))node.setAttribute(key,String(value));
      if(text!==undefined)node.textContent=text;
      nodes.push(node);return node;
    }
    add("title",{},"五步 Muon 与 Pion 的谱响应；标记表示当前四个归一化输入，完整系数见结果表");
    for(const value of yTicks) {
      add("line",{x1:left,x2:left+plotWidth,y1:y(value),y2:y(value),class:"pion-spectrum__grid"});
      add("text",{x:left-8,y:y(value),"text-anchor":"end","dominant-baseline":"middle"},String(value));
    }
    for(const value of plotWidth < font*8 ? [0,1] : [0,.5,1]) {
      add("line",{x1:x(value),x2:x(value),y1:top,y2:top+plotHeight,class:"pion-spectrum__grid"});
      add("text",{x:x(value),y:top+plotHeight+font*1.5,"text-anchor":value===0?"start":value===1?"end":"middle"},String(value));
    }
    for(const name of ["identity","muon","pion"]) {
      const pairs=Array.from({length:801},(_,index) => {
        const input=index/800,output=name==="identity"?input:response(input,result.promotion_steps,name==="muon");
        return (index?"L":"M")+x(input).toFixed(3)+" "+y(output).toFixed(3);
      });
      add("path",{d:pairs.join(" "),class:"pion-spectrum__curve pion-spectrum__curve--"+name});
    }
    for(let index=0;index<result.normalized.length;index++) {
      const px=x(result.normalized[index]);
      add("rect",{x:px-3,y:y(result.muon[index])-3,width:6,height:6,class:"pion-spectrum__marker pion-spectrum__marker--muon"});
      add("circle",{cx:px,cy:y(result.pion[index]),r:3.5,class:"pion-spectrum__marker pion-spectrum__marker--pion"});
    }
    svg.setAttribute("viewBox","0 0 "+width+" "+height);svg.style.height=height+"px";svg.replaceChildren(...nodes);
  }
  function mount(element) {
    if(element.dataset.ready)return;
    const controls=element.querySelector("[data-controls]"),form=controls.querySelector("form"),
      plot=element.querySelector("[data-chart]"),svg=plot.querySelector("svg"),error=element.querySelector("[data-error]"),
      note=element.querySelector("[data-note]"),staticNote=element.querySelector("[data-static]");
    let lastResult=null,lastWidth=0,lastFont=0,lastSpacing="",printing=false;
    const numbers=[...form.querySelectorAll('input[type="number"]')];
    function read() {
      let invalid=null;
      for(const input of numbers) {
        const bad=!input.checkValidity()||!Number.isFinite(input.valueAsNumber);
        input.setAttribute("aria-invalid",String(bad));if(bad&&!invalid)invalid=input;
      }
      const choice=form.elements.promotionSteps.value,kp=choice===""?NaN:Number(choice),badSteps=!Number.isInteger(kp)||kp<0||kp>5;
      form.elements.promotionSteps.setAttribute("aria-invalid",String(badSteps));
      if(badSteps&&!invalid)invalid=form.elements.promotionSteps;
      return {invalid,raw:numbers.map(input=>input.valueAsNumber),kp,epsilon:form.elements.includeEpsilon.checked?1e-7:0};
    }
    function write(result) {
      for(const name of ["input_erank","pion_erank","muon_norm","pion_norm"])writeNumber(element.querySelector('[data-value="'+name+'"]'),result?result[name]:null);
      for(const name of ["normalized","muon","pion"])for(const [index,node]of [...element.querySelectorAll('[data-series="'+name+'"]')].entries())writeNumber(node,result?result[name][index]:null);
      if(!result){note.textContent="修正输入后重新计算；不显示上一次曲线或结果。";return;}
      note.textContent=result.all_zero?"原始输入全为零：没有生成非零方向；有效秩按本文约定记为 0。":
        "当前 "+result.promotion_steps+" 次提升、"+result.suppression_steps+" 次抑制，ε = "+(result.epsilon?"10⁻⁷":"0")+"。"+
        (result.pion_norm<1e-12?"Pion 输出幅度很小，即使有效秩仍可能很高。":"有效秩描述谱形状，输出范数描述幅度。")+
        "小系数不自动代表噪声，显示为零也可能涉及浮点下溢。";
    }
    function update() {
      const values=read();let result=null;
      if(!values.invalid)try{result=calculate(values.raw,values.kp,values.epsilon);}catch(_){/* Show a recoverable input error. */}
      error.hidden=!!result;
      error.textContent=result?"":values.invalid?values.invalid.labels[0].textContent.trim()+"：请填写允许范围内的有限数值。":"无法计算，请检查谱幅度与提升次数。";
      lastResult=result;write(result);plot.hidden=!result;
      if(result)try{draw(svg,result);}catch(_){chartFailure();}
    }
    function chartFailure() {
      plot.hidden=true;error.hidden=false;error.textContent="曲线无法绘制；当前数值结果仍可查看，调整输入可重试。";
    }
    // Keep the native default table/cards intact if the initial SVG setup fails.
    const initial=read(),result=calculate(initial.raw,initial.kp,initial.epsilon);
    plot.hidden=false;draw(svg,result);lastResult=result;write(result);
    controls.hidden=false;staticNote.hidden=true;element.dataset.ready="true";
    form.addEventListener("submit",event=>{event.preventDefault();update();});
    form.addEventListener("input",update);form.addEventListener("change",update);
    for(const button of controls.querySelectorAll("[data-preset]"))button.addEventListener("click",()=>{
      const preset=presets[button.dataset.preset];
      numbers.forEach((input,index)=>{input.value=String(preset.raw[index]);});
      form.elements.promotionSteps.value=String(preset.kp);form.elements.includeEpsilon.checked=preset.epsilon;update();
    });
    function redraw(force=false) {
      if(printing||!lastResult||plot.hidden)return;
      const style=getComputedStyle(svg),width=svg.getBoundingClientRect().width,font=parseFloat(style.fontSize),spacing=style.letterSpacing+":"+style.wordSpacing;
      if(force===true||Math.abs(width-lastWidth)>.5||Math.abs(font-lastFont)>.1||spacing!==lastSpacing) {
        lastWidth=width;lastFont=font;lastSpacing=spacing;
        try{draw(svg,lastResult);}catch(_){chartFailure();}
      }
    }
    if(typeof ResizeObserver==="function")new ResizeObserver(redraw).observe(plot);else window.addEventListener("resize",redraw);
    if(typeof MutationObserver==="function")new MutationObserver(redraw).observe(document.documentElement,{attributes:true,attributeFilter:["style","class"]});
    document.fonts?.ready.then(redraw);
    window.addEventListener("beforeprint",()=>{
      printing=true;
      // Native print events can precede print-media layout. Use the same fixed
      // coordinate width as the CSS print cap, then restore screen coordinates.
      if(lastResult&&!plot.hidden)try {
        const width=parseFloat(getComputedStyle(element).getPropertyValue("--pion-print-width"));
        draw(svg,lastResult,Number.isFinite(width)&&width>0?width:undefined);
      }catch(_){chartFailure();}
    });
    window.addEventListener("afterprint",()=>{printing=false;redraw(true);});
  }
  if(typeof module!=="undefined"&&module.exports)module.exports={calculate,normalize,effectiveRank,response,format,presets};
  if(typeof document!=="undefined") {
    const initialize=()=>document.querySelectorAll("[data-pion-spectrum]").forEach(element=>{
      try{mount(element);}catch(_){element.querySelector("[data-controls]").hidden=true;element.querySelector("[data-chart]").hidden=true;element.querySelector("[data-static]").hidden=false;}
    });
    if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",initialize,{once:true});else initialize();
  }
})();
