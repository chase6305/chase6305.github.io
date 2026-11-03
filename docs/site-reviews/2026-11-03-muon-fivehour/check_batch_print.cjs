// Native PDFs of the component within the full article. Serial private CDP only.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {connect} = require('../../../scripts/lib/browser.cjs');
const {contrastAudit} = require('../../../scripts/lib/contrast.cjs');
const [base, portText, out] = process.argv.slice(2);
const cases = [
  {name:'valid', scripts:true, theme:'dark'},
  {name:'pending', scripts:true, theme:'warm'},
  {name:'invalid', scripts:true, theme:'light'},
  {name:'native', scripts:false, theme:'warm'},
  {name:'forced', scripts:true, theme:'dark', forced:true},
];
(async()=>{
  fs.mkdirSync(out,{recursive:true});
  const b=await connect(Number(portText)), report={cases:[],exceptions:b.exceptions};
  const run=(fn,...args)=>b.evaluate('('+fn.toString()+')(...'+JSON.stringify(args)+')');
  const snapshot=()=>run(()=>{
    const lab=document.querySelector('[data-muon-batch]');
    return {values:[...lab.querySelectorAll('input')].map(i=>[i.name,i.value,i.getAttribute('aria-invalid')]),
      outputs:[...lab.querySelectorAll('output')].map(o=>[o.dataset.value,o.textContent]),
      error:lab.querySelector('[data-error]').textContent,status:lab.querySelector('[data-status]').textContent,
      hash:location.hash,focus:document.activeElement.id,top:document.activeElement.getBoundingClientRect().top,
      open:[...document.querySelectorAll('.content details')].map(d=>d.open)};
  });
  try{
    for(const item of cases) for(const size of [100,200]){
      const features=item.forced?[{name:'forced-colors',value:'active'},{name:'prefers-color-scheme',value:'dark'}]:[];
      await b.cdp('Emulation.setScriptExecutionDisabled',{value:!item.scripts});
      await b.cdp('Emulation.setEmulatedMedia',{media:'',features});
      await b.viewport(320,1000);await b.navigate('about:blank');
      await b.navigate(base+'/posts/ai/muon-learning-rate-batch-size/#batch-calculator');
      await run(async(size,theme)=>{
        document.documentElement.classList.remove('dark','warm');
        if(theme!=='light')document.documentElement.classList.add(theme);
        document.documentElement.style.colorScheme=theme==='dark'?'dark':'light';
        document.documentElement.style.fontSize=size+'%';
        await document.fonts.ready;
        await Promise.all([...document.querySelectorAll('.content img')].map(i=>{i.loading='eager';return i.decode()}));
      },size,item.theme);
      await run(name=>{
        const lab=document.querySelector('[data-muon-batch]'),f=lab.querySelector('form');
        if(name==='valid'||name==='forced'){
          lab.querySelector('[data-preset=smaller]').click();
          f.elements.batchNew.focus();
        }else if(name==='pending'||name==='invalid'){
          f.elements.batchNew.value=name==='invalid'?'0':'2048';
          f.elements.batchNew.focus();
          f.elements.batchNew.dispatchEvent(new Event('input',{bubbles:true}));
          if(name==='invalid')f.requestSubmit();
        }else{
          const link=document.querySelector('.content a[href="batch_math.py"]');link.focus();
        }
        document.activeElement.scrollIntoView({block:'center',behavior:'instant'});
      },item.name);
      await b.pause(150);
      const cards=await run(()=>[...document.querySelectorAll('.muon-batch__results > div')].map(e=>({
        label:e.querySelector('dt').textContent,value:e.querySelector('output').textContent})));
      assert.equal(cards.length,7);
      await b.cdp('Emulation.setEmulatedMedia',{media:'print',features});
      const palette=await run(()=>{
        const lab=document.querySelector('[data-muon-batch]'),s=getComputedStyle(lab);
        return {color:s.color,background:s.backgroundColor,scheme:getComputedStyle(document.documentElement).colorScheme};
      });
      const text=await run(contrastAudit,'.muon-batch');
      assert.equal(palette.background,'rgb(255, 255, 255)',JSON.stringify({item,palette}));
      assert.equal(palette.scheme,'light');assert.deepEqual(text.issues,[]);
      await b.cdp('Emulation.setEmulatedMedia',{media:'',features});
      await b.pause(120);
      // CDP media emulation changes CSS before its change event; native print
      // has beforeprint first. Start the native restoration check on screen.
      await run(()=>document.activeElement.scrollIntoView({block:'center',behavior:'instant'}));
      await b.pause(120);const before=await snapshot();
      for(const paper of [{name:'a4',width:8.27,height:11.69},{name:'letter',width:8.5,height:11}]){
        const file=path.join(out,[item.name,size,paper.name].join('-')+'.pdf');
        const pdf=await b.cdp('Page.printToPDF',{printBackground:true,paperWidth:paper.width,paperHeight:paper.height,
          marginTop:.4,marginBottom:.4,marginLeft:.4,marginRight:.4,generateTaggedPDF:true});
        fs.writeFileSync(file,Buffer.from(pdf.data,'base64'));
        const xml=execFileSync('pdftotext',['-bbox',file,'-'],{encoding:'utf8',maxBuffer:16*1024*1024,timeout:20000});
        const checked=JSON.parse(execFileSync('python3',['-B',path.join(__dirname,'inspect_pdf_cards.py'),JSON.stringify(cards)],
          {input:xml,encoding:'utf8',maxBuffer:1024*1024,timeout:30000}));
        await b.pause(200);const after=await snapshot();
        assert.deepEqual({...after,top:0},{...before,top:0},'Print changed current calculation or focus');
        assert(Math.abs(after.top-before.top)<=1,JSON.stringify({before,after}));
        report.cases.push({name:item.name,size,theme:item.theme,forced:!!item.forced,scripts:item.scripts,paper:paper.name,
          file,palette,textSamples:text.checked,restored:true,...checked});
        assert(checked.passed,JSON.stringify(report.cases.at(-1)));
      }
    }
    assert.deepEqual(b.exceptions,[]);
    console.log(JSON.stringify({pdfs:report.cases.length,cardPairs:report.cases.reduce((sum,c)=>sum+c.cards.length,0),exceptions:0}));
  }finally{
    fs.writeFileSync(path.join(out,'results.json'),JSON.stringify(report,null,2)+'\n');
    await b.cdp('Emulation.setScriptExecutionDisabled',{value:false}).catch(()=>{});
    await b.cdp('Emulation.setEmulatedMedia',{media:'',features:[]}).catch(()=>{});await b.close();
  }
})().catch(e=>{console.error(e);process.exitCode=1});
