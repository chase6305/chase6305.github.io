(function () {
  'use strict';
  const finite = (...values) => values.every(Number.isFinite);
  function timing(horizon=16, stride=5, period=20, observation=0, inference=80, execution=0) {
    if (!finite(horizon,stride,period,observation,inference,execution) ||
        !Number.isInteger(horizon) || horizon<8 || horizon>64 ||
        !Number.isInteger(stride) || stride<1 || stride>horizon || period<1 || period>100 ||
        [observation,inference,execution].some(v=>v<0||v>1000) ||
        [period,observation,inference,execution].some(v=>Math.abs(v*10-Math.round(v*10))>1e-9)) {
      throw new RangeError('块长、执行间隔或时间参数超出范围；时间使用 0.1 ms 的整数倍。');
    }
    const totalTicks = [observation,inference,execution].reduce((sum,v)=>sum+Math.round(v*10),0);
    const periodTicks = Math.round(period*10);
    const delay = Math.floor((totalTicks+periodTicks-1)/periodTicks);
    const workDelay = Math.floor((Math.round(inference*10)+periodTicks-1)/periodTicks);
    const overlap = horizon-stride;
    const schedulable = workDelay<=stride && stride<=horizon-delay;
    const weights = schedulable ? Array.from({length:horizon},(_,i)=>{
      if(i<delay) return 1;
      if(i>=overlap) return 0;
      const c=(overlap-i)/(overlap-delay+1);
      return c*Math.expm1(c)/Math.expm1(1);
    }) : [];
    return {horizon,stride,delay,workDelay,totalDelay:totalTicks/10,remaining:overlap,schedulable,
      frozen:schedulable?delay:0,editable:schedulable?overlap-delay:0,future:schedulable?stride:0,
      noStride:workDelay+delay>horizon,weights};
  }
  function residual(gain=1, feedback=0, correction=-.2, bias=.2, period=.1) {
    if (!finite(gain,feedback,correction,bias,period) || gain<.1||gain>3 || feedback< -1||feedback>4 ||
        correction< -1||correction>1 || bias< -.5||bias>.5 || period<.01||period>1) {
      throw new RangeError('反馈、偏置或采样参数超出允许范围。');
    }
    const actual=[1], baseline=[1], steps=60;
    for (let i=0;i<steps;i++) {
      actual.push(actual[i]+period*((feedback-gain)*actual[i]+correction+bias));
      baseline.push(baseline[i]+period*(-gain*baseline[i]+bias));
    }
    const pole=1+period*(feedback-gain), stable=Math.abs(pole)<1;
    return {pole,stable,forcing:correction+bias,finalError:actual[steps],duration:steps*period,
      fixedPoint:stable?(correction+bias)/(gain-feedback):null,actual,baseline,steps};
  }
  function posterior(sigma=2, angle=90, second=false) {
    if (!finite(sigma,angle) || sigma<1||sigma>8 || angle<0||angle>90 || typeof second!=='boolean') {
      throw new RangeError('距离尺度为 1 到 8 mm，法向夹角为 0° 到 90°。');
    }
    const theta=angle*Math.PI/180, c=angle===90?0:Math.cos(theta), s=angle===0?0:Math.sin(theta);
    const points=[], logs=[], axis=Array.from({length:41},(_,i)=>i-20);
    for (const x of axis) for (const y of axis) {
      const first=y+4, other=c*first+s*(x-6);
      points.push([x,y]);logs.push(-(first*first+(second?other*other:0))/(2*sigma*sigma));
    }
    const max=Math.max(...logs), values=logs.map(v=>Math.exp(v-max)), total=values.reduce((a,b)=>a+b,0);
    const weights=values.map(v=>v/total), mean=[0,0], variance=[0,0], marginals=[Array(41).fill(0),Array(41).fill(0)];
    points.forEach((p,i)=>{mean[0]+=p[0]*weights[i];mean[1]+=p[1]*weights[i]});
    let covariance=0, squares=0;
    points.forEach((p,i)=>{
      const w=weights[i], dx=p[0]-mean[0], dy=p[1]-mean[1];
      variance[0]+=w*dx*dx;variance[1]+=w*dy*dy;covariance+=w*dx*dy;squares+=w*w;
      marginals[0][p[0]+20]+=w;marginals[1][p[1]+20]+=w;
    });
    return {meanX:mean[0],meanY:mean[1],stdX:Math.sqrt(variance[0]),stdY:Math.sqrt(variance[1]),
      correlation:covariance/Math.sqrt(variance[0]*variance[1]),effectiveSamples:1/squares,axis,marginals};
  }
  if (typeof module!=='undefined'&&module.exports) module.exports={timing,residual,posterior};
  if (typeof document==='undefined') return;
  const NS='http://www.w3.org/2000/svg';
  const format = v => Math.abs(v)>=1e6?v.toExponential(3):(Math.abs(v)<.00005?0:v).toFixed(4);
  function svgElement(name,attributes) {
    const node=document.createElementNS(NS,name);
    Object.entries(attributes).forEach(([k,v])=>node.setAttribute(k,String(v)));
    return node;
  }
  function svgPath(points,className) {
    return svgElement('path',{d:points.map(([x,y],i)=>(i?'L':'M')+x.toFixed(5)+','+y.toFixed(5)).join(' '),class:className});
  }
  function axes(fragment,zero=150) {
    fragment.append(svgElement('line',{x1:20,y1:zero,x2:380,y2:zero,class:'deployment-lab__axis'}));
  }
  function plotTiming(model) {
    const fragment=document.createDocumentFragment();
    if (!model.schedulable) return {fragment,axis:'',label:'当前时间预算不满足所列串行调度条件。'};
    const groups=[[0,model.delay,'one'],[model.delay,model.horizon-model.stride,'two'],[model.horizon-model.stride,model.horizon,'three']];
    groups.forEach(([a,b,name])=>{
      if(b>a) fragment.append(svgElement('rect',{x:20+a/model.horizon*360,y:8,width:(b-a)/model.horizon*360,height:29,rx:2,class:'deployment-lab__band-'+name}));
    });
    axes(fragment);
    fragment.append(svgPath(model.weights.map((w,i)=>[20+i/Math.max(1,model.horizon-1)*360,150-90*w]),'deployment-lab__curve-one'));
    return {fragment,axis:`上方为三个动作区间；下方为引导权重 0 至 1。横轴为新块索引 0 至 ${model.horizon-1}。`,
      label:`已承诺 ${model.frozen} 步，可修改 ${model.editable} 步，新未来 ${model.future} 步；下方显示对应软掩码。`};
  }
  function plotResidual(model) {
    const fragment=document.createDocumentFragment();
    const extent=Math.max(...model.actual.map(Math.abs),...model.baseline.map(Math.abs),1)*1.05;
    const y=v=>85-v/extent*66;
    axes(fragment,y(0));
    [model.actual,model.baseline].forEach((values,i)=>fragment.append(svgPath(values.map((v,j)=>[20+j/model.steps*360,y(v)]),i?'deployment-lab__curve-two':'deployment-lab__curve-one')));
    return {fragment,axis:`横轴 0 至 ${format(model.duration)} s；纵轴误差范围为 ±${format(extent)}。曲线连接离散采样点。`,
      label:`当前残差的第 60 步误差为 ${format(model.finalError)}，零残差为 ${format(model.baseline[60])}；曲线为离散标量系统。`};
  }
  function plotPosterior(model) {
    const fragment=document.createDocumentFragment();
    const maximum=Math.max(...model.marginals.flat())*1.1;
    axes(fragment);
    model.marginals.forEach((values,i)=>fragment.append(svgPath(values.map((v,j)=>[20+j/40*360,150-v/maximum*138]),i?'deployment-lab__curve-two':'deployment-lab__curve-one')));
    return {fragment,axis:`横轴平移 −20 至 20 mm，每格 1 mm；纵轴为每格离散质量，范围 0 至 ${format(maximum)}，两条曲线共用刻度。`,
      label:`x 平移标准差 ${format(model.stdX)} mm，y 平移标准差 ${format(model.stdY)} mm；实线和虚线分别为两维边缘质量。`};
  }
  const configurations={
    timing:{keys:['horizon','stride','period','observation','inference','execution'],calculate:timing,plot:plotTiming,
      presets:{default:[16,5,20,0,80,0],latencies:[16,6,20,40,60,20],short:[16,3,20,0,80,0],exhausted:[16,5,20,0,240,0]},
      values:m=>({...m,schedule:m.schedulable?'满足':'不满足',regions:m.schedulable?`${m.frozen}／${m.editable}／${m.future}`:'不适用'}),
      note:m=>m.schedulable?`当前满足 d_work ≤ s ≤ H − d_deadline。后台计算占 ${m.workDelay} 步，按总偏移已承诺的前缀占 ${m.delay} 步；两者仅在其他时延为零时相同。`:
        (m.noStride?'当前 d_work＋d_deadline＞H，在此模型下不存在可行的串行执行间隔。':'当前执行间隔不满足条件。')+
        (m.workDelay>m.stride?' 后台计算占用步数大于 s。':'')+(m.delay>m.remaining?' 旧块在新结果能够执行前已用尽。':'')},
    residual:{keys:['gain','feedback','correction','bias','period'],calculate:residual,plot:plotResidual,
      presets:{default:[1,0,-.2,.2,.1],baseline:[1,0,0,.2,.1],wrong:[1,1.2,0,0,.1],sampled:[3,0,0,0,.8]},
      values:m=>({...m,stable:m.stable?'满足 |p|＜1':'不满足 |p|＜1',fixedPoint:m.fixedPoint===null?'不适用':m.fixedPoint}),
      note:m=>m.stable?'当前离散系统收敛到所列定点。常量残差抵消偏差时定点可为零；这不是一般机器人残差控制的稳定性证明。':
        '当前不满足离散收缩条件。错误反馈方向或过大的采样周期都可能破坏收敛；这里不使用限幅掩盖轨迹增长。'},
    contact:{keys:['sigma','angle'],calculate:posterior,plot:plotPosterior,
      presets:{default:[2,90,false],parallel:[2,0,true],oblique:[2,15,true],orthogonal:[2,90,true],broad:[8,90,true]},
      values:m=>m,note:(_m,second)=>second?'增加独立方向可减少相应平移歧义；平行表面主要进一步收紧同一个方向。有限网格、噪声模型和已知关联共同决定这里的后验。':
        '只接触地面时，x 平移仍服从均匀先验。y 已变得集中，不能据此认定两个方向都已经标定。'}
  };
  function setup(root) {
    if(root.dataset.ready) return;
    const kind=root.dataset.deploymentLab, config=configurations[kind];
    if(!config) return;
    const form=root.querySelector('form'), fields=config.keys.map(k=>form.elements.namedItem(k));
    const outputs=[...root.querySelectorAll('[data-value]')], error=root.querySelector('[data-error]');
    const computed=root.querySelector('[data-computed]'), plot=root.querySelector('[data-plot]'), chart=root.querySelector('[data-chart]');
    const axis=root.querySelector('[data-axis]'), note=root.querySelector('[data-note]');
    function clear(message,invalid) {
      fields.forEach(f=>invalid.includes(f)?f.setAttribute('aria-invalid','true'):f.removeAttribute('aria-invalid'));
      if(error.textContent!==message) error.textContent=message;
      error.hidden=false;computed.hidden=true;
      outputs.forEach(o=>o.textContent='');
      // Error cleanup must still work if an SVG update method itself failed.
      while(chart.firstChild) chart.removeChild(chart.firstChild);
      axis.textContent='';note.textContent='';
    }
    function render(initial=false) {
      const invalid=fields.filter(f=>!f.validity.valid||!Number.isFinite(f.valueAsNumber));
      if(invalid.length) { clear('请填写各字段标示范围内的有限数值，并遵守整数或步长要求。',invalid);return; }
      try {
        const second=kind==='contact'?form.elements.namedItem('second').checked:undefined;
        const args=fields.map(f=>f.valueAsNumber);if(kind==='contact') args.push(second);
        const model=config.calculate(...args), drawing=config.plot(model), values=config.values(model);
        // Build SVG off-document before changing the visible result state.
        chart.replaceChildren(drawing.fragment);chart.setAttribute('aria-label',drawing.label);
        outputs.forEach(o=>{
          const v=values[o.dataset.value], text=typeof v==='number'?(kind==='timing'&&['delay','workDelay','remaining'].includes(o.dataset.value)?String(v):format(v)):String(v);
          if(o.textContent!==text) o.textContent=text;
        });
        fields.forEach(f=>f.removeAttribute('aria-invalid'));
        axis.textContent=drawing.axis;note.textContent=config.note(model,second);
        plot.hidden=kind==='timing'&&!model.schedulable;
        computed.hidden=false;error.hidden=true;error.textContent='';
      } catch(e) {
        if(initial) throw e;
        clear(e instanceof RangeError?e.message:'结果暂时无法更新；请重新选择参数或预设。',e instanceof RangeError?fields:[]);
      }
    }
    try { render(true); } catch(_e) { return; }
    form.addEventListener('submit',event=>event.preventDefault());
    form.addEventListener('input',()=>render());
    root.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>{
      const values=config.presets[button.dataset.preset];
      fields.forEach((field,i)=>field.value=String(values[i]));
      if(kind==='contact') form.elements.namedItem('second').checked=values[fields.length];
      render();
    }));
    root.dataset.ready='true';root.querySelector('[data-interactive]').hidden=false;root.querySelector('[data-static-result]').hidden=true;
  }
  document.querySelectorAll('[data-deployment-lab]').forEach(setup);
})();
