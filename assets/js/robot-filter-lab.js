/* Deterministic scalar filtering and progressive enhancement for the article. */
(function () {
  "use strict";
  const defaults = {method: "oneEuro", noise: 1, frequency: 1, beta: 4, window: 4, damping: 1};
  const methodFrequency = {oneEuro: 1, butterworth: 3, wma: 1, ema: 3, msd: 5};
  const notes = {
    oneEuro: "变化率使用当前输入与前一滤波输出的差；导数截止频率固定为 1 Hz。β 的单位随位置单位变化。",
    butterworth: "二阶数字 Butterworth，截止频率为 −3 dB 点。保留递归状态，以第一条测量的常值稳态初始化。",
    wma: "从最新到最旧使用 N、N−1、…、1 的归一化权重。未满窗口时重新归一化，重复值仍占一个采样时刻。",
    ema: "一阶低通系数为 dt / (dt + 1 / (2πfc))。这是后向 Euler 形式，数字 −3 dB 点不必恰好等于输入参数。",
    msd: "频率参数是自然频率，不是通用的 −3 dB 截止频率。精确推进零阶保持的二阶状态，前一目标作用于当前采样区间。"
  };
  const alpha = (f, dt) => dt / (dt + 1 / (2 * Math.PI * f));

  function validateData(data) {
    if (!data || !Number.isFinite(data.sample_hz) || data.sample_hz <= 0 ||
        !Array.isArray(data.t) || data.t.length < 3 || data.t.length > 100000 ||
        !Array.isArray(data.truth) || !Array.isArray(data.measurement) ||
        data.truth.length !== data.t.length || data.measurement.length !== data.t.length) {
      throw new RangeError("Invalid signal data");
    }
    for (let i = 0; i < data.t.length; i++) {
      if (![data.t[i], data.truth[i], data.measurement[i]].every(Number.isFinite) ||
          (i && Math.abs(data.t[i] - data.t[i-1] - 1/data.sample_hz) > 1e-8)) {
        throw new RangeError("Expected finite uniformly sampled data");
      }
    }
  }

  function msdTransition(frequency, damping, dt) {
    const w = 2 * Math.PI * frequency;
    const exponential = Math.exp(-damping * w * dt);
    let cosine, sine;
    if (Math.abs(damping - 1) < 1e-8) { cosine = 1; sine = dt; }
    else if (damping < 1) {
      const d = w * Math.sqrt(1 - damping*damping);
      cosine = Math.cos(d*dt); sine = Math.sin(d*dt)/d;
    } else {
      const d = w * Math.sqrt(damping*damping - 1);
      cosine = Math.cosh(d*dt); sine = Math.sinh(d*dt)/d;
    }
    const a00 = exponential*(cosine+damping*w*sine);
    const a01 = exponential*sine;
    const a10 = -exponential*w*w*sine;
    const a11 = exponential*(cosine-damping*w*sine);
    return [a00, a01, a10, a11, 1-a00, -a10];
  }

  function calculate(data, settings) {
    validateData(data);
    const s = {...defaults, ...settings};
    if (!Object.hasOwn(notes, s.method) ||
        ![s.noise,s.frequency,s.beta,s.window,s.damping].every(Number.isFinite) ||
        s.noise < 0 || s.noise > 2 || s.frequency < .2 || s.frequency > 20 ||
        s.frequency >= data.sample_hz/2 || s.beta < 0 || s.beta > 20 ||
        !Number.isInteger(s.window) || s.window < 1 || s.window > 31 ||
        s.damping < .3 || s.damping > 2) throw new RangeError("Invalid filter parameters");
    const x = data.measurement.map((v,i) => data.truth[i] + s.noise*(v-data.truth[i]));
    const y = new Array(x.length), dt = 1/data.sample_hz;
    let value = x[0], derivative = 0, velocity = 0;
    let px1 = x[0], px2 = x[0], py1 = x[0], py2 = x[0];
    const k = Math.tan(Math.PI*s.frequency/data.sample_hz);
    const norm = 1/(1 + Math.SQRT2*k + k*k);
    const b0 = k*k*norm, b1 = 2*b0, b2 = b0;
    const a1 = 2*(k*k-1)*norm, a2 = (1-Math.SQRT2*k+k*k)*norm;
    const transition = msdTransition(s.frequency,s.damping,dt);
    for (let i = 0; i < x.length; i++) {
      if (s.method === "wma") {
        let sum = 0, total = 0;
        for (let age = 0; age < Math.min(i+1,s.window); age++) {
          const weight = s.window-age; sum += weight*x[i-age]; total += weight;
        }
        value = sum/total;
      } else if (s.method === "butterworth") {
        value = b0*x[i] + b1*px1 + b2*px2 - a1*py1 - a2*py2;
        px2 = px1; px1 = x[i]; py2 = py1; py1 = value;
      } else if (i && s.method === "msd") {
        const [a,b,c,d,e,f] = transition;
        const next = a*value + b*velocity + e*x[i-1];
        velocity = c*value + d*velocity + f*x[i-1]; value = next;
      } else if (i && (s.method === "oneEuro" || s.method === "ema")) {
        const ad = alpha(1,dt);
        derivative = ad*(x[i]-value)/dt + (1-ad)*derivative;
        const cutoff = s.frequency + (s.method === "oneEuro" ? s.beta*Math.abs(derivative) : 0);
        const a = alpha(cutoff,dt); value = a*x[i] + (1-a)*value;
      }
      y[i] = value;
    }
    const still = [], lag = [];
    let squared = 0;
    for (let i = 0; i < y.length; i++) {
      if (data.t[i] >= .8 && data.t[i] < 1.8) still.push(y[i]);
      if (data.t[i] >= 2.5 && data.t[i] < 3.5) lag.push((data.truth[i]-y[i])/.2);
      squared += (data.truth[i]-y[i])**2;
    }
    if (!still.length || !lag.length) throw new RangeError("Signal misses metric windows");
    const mean = still.reduce((a,b)=>a+b,0)/still.length;
    return {measurement:x, output:y,
      jitter:Math.sqrt(still.reduce((a,b)=>a+(b-mean)**2,0)/still.length)*1000,
      lag:lag.reduce((a,b)=>a+b,0)/lag.length*1000,
      rmse:Math.sqrt(squared/y.length)*1000};
  }

  function draw(canvas, data, result, view) {
    const width = canvas.clientWidth, height = canvas.clientHeight;
    if (!width || !height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
    const ctx = canvas.getContext("2d"); if (!ctx) return;
    ctx.scale(ratio,ratio);
    const dark = document.documentElement.classList.contains("dark");
    const colors = {grid:dark?"#354458":"#dce4ed",text:dark?"#cbd8e8":"#465a70",raw:dark?"#6d7d90":"#bcc5cf",truth:dark?"#eef3f9":"#26384c",filtered:dark?"#86bdff":"#317ebd"};
    const ranges = {all:[0,10],still:[.8,1.8],onset:[2,2.65]};
    const [lo,hi] = ranges[view];
    const indexes = data.t.map((_,i)=>i).filter(i=>data.t[i]>=lo&&data.t[i]<=hi);
    let min = Infinity,max = -Infinity;
    for (const i of indexes) for (const v of [data.truth[i],result.measurement[i],result.output[i]]) {min=Math.min(min,v*1000);max=Math.max(max,v*1000);}
    const pad = Math.max(2,(max-min)*.12);min-=pad;max+=pad;
    const left=49,right=12,top=20,bottom=35,w=width-left-right,h=height-top-bottom;
    const tx=t=>left+(t-lo)/(hi-lo)*w, ty=v=>top+(max-v)/(max-min)*h;
    ctx.font="11px system-ui, sans-serif";ctx.fillStyle=colors.text;
    for(let i=0;i<=4;i++){
      const v=min+(max-min)*i/4, yy=ty(v);
      ctx.strokeStyle=colors.grid;ctx.lineWidth=.6;ctx.beginPath();ctx.moveTo(left,yy);ctx.lineTo(width-right,yy);ctx.stroke();
      ctx.textAlign="right";ctx.fillText(v.toFixed(0),left-6,yy+4);
    }
    for(let i=0;i<=2;i++){
      const v=lo+(hi-lo)*i/2;ctx.textAlign="center";ctx.fillText(v.toFixed(view==='all'?0:2),tx(v),height-16);
    }
    ctx.textAlign="left";ctx.fillText("mm",5,12);ctx.textAlign="right";ctx.fillText("s",width-1,height-16);
    function line(values,color,lineWidth,dash){
      ctx.save();ctx.beginPath();ctx.rect(left,top,w,h);ctx.clip();ctx.strokeStyle=color;ctx.lineWidth=lineWidth;ctx.setLineDash(dash);ctx.beginPath();
      indexes.forEach((i,j)=>{const x=tx(data.t[i]),y=ty(values[i]*1000);j?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.restore();
    }
    line(result.measurement,colors.raw,.8,[]);line(data.truth,colors.truth,1.8,[5,4]);line(result.output,colors.filtered,2.2,[]);
    canvas.setAttribute("aria-label",`合成位置曲线，${lo} 至 ${hi} 秒。静止标准差 ${result.jitter.toFixed(3)} 毫米，匀速等效滞后 ${result.lag.toFixed(3)} 毫秒。`);
  }

  async function mount(element) {
    if (element.dataset.ready) return; element.dataset.ready="true";
    const status=element.querySelector("[data-filter-status]");status.hidden=false;
    try {
      const response=await fetch(element.dataset.signal);if(!response.ok)throw new Error("Signal unavailable");
      const data=await response.json();validateData(data);
      const form=element.querySelector("form"), canvas=element.querySelector("canvas");
      const error=element.querySelector("[data-filter-error]");let result,view="all";
      function update(){
        const values={};for(const input of form.elements)if(input.name)values[input.name]=input.name==="method"?input.value:Number(input.value);
        for(const label of element.querySelectorAll("[data-parameter]")){
          const name=label.dataset.parameter;
          label.hidden=(name==="frequency"&&values.method==="wma")||(name==="beta"&&values.method!=="oneEuro")||(name==="window"&&values.method!=="wma")||(name==="damping"&&values.method!=="msd");
        }
        element.querySelector("[data-frequency-label]").textContent=values.method==="msd"?"自然频率（Hz）":values.method==="oneEuro"?"最低截止频率（Hz）":"截止参数（Hz）";
        for(const output of element.querySelectorAll("[data-control]")){const key=output.dataset.control;output.textContent=values[key].toFixed(key==="window"?0:key==="damping"?2:1);}
        try{
          result=calculate(data,values);error.hidden=true;
          for(const name of ["jitter","lag","rmse"])element.querySelector(`[data-metric="${name}"]`).textContent=result[name].toFixed(3)+(name==="lag"?" ms":" mm");
          element.querySelector("[data-filter-note]").textContent=notes[values.method];draw(canvas,data,result,view);
        }catch(_){result=null;error.hidden=false;error.textContent="参数无效，请恢复示例参数后重试。";for(const output of element.querySelectorAll("[data-metric]"))output.textContent="—";const ctx=canvas.getContext("2d");ctx?.clearRect(0,0,canvas.width,canvas.height);canvas.setAttribute("aria-label","参数无效，当前没有可用的滤波曲线。");}
      }
      form.addEventListener("submit",event=>event.preventDefault());form.addEventListener("input",event=>{if(event.target.name==="method")form.elements.frequency.value=methodFrequency[event.target.value];update()});
      for(const button of element.querySelectorAll("[data-view]"))button.addEventListener("click",()=>{view=button.dataset.view;element.querySelectorAll("[data-view]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));if(result)draw(canvas,data,result,view)});
      element.querySelector("[data-filter-reset]").addEventListener("click",()=>{for(const [key,value] of Object.entries(defaults))form.elements.namedItem(key).value=value;update()});
      const redraw=()=>{if(result)draw(canvas,data,result,view)};
      new ResizeObserver(redraw).observe(canvas);new MutationObserver(redraw).observe(document.documentElement,{attributes:true,attributeFilter:["class"]});
      element.querySelector("[data-filter-interactive]").hidden=false;status.hidden=true;update();
    }catch(_){status.textContent="交互信号暂时无法载入，请使用下方静态曲线或下载脚本。";}
  }
  if(typeof module!=="undefined"&&module.exports)module.exports={calculate,msdTransition,defaults};
  if(typeof document!=="undefined"){
    const initialize=()=>document.querySelectorAll("[data-robot-filter-lab]").forEach(mount);
    if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",initialize,{once:true});else initialize();
  }
})();
