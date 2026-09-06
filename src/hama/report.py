"""Build a self-contained HTML report from HAMA checkpoints."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def build_training_report(
    output: str | Path,
    run_dirs: list[str | Path],
    *,
    title: str = "HAMA training record",
) -> Path:
    """Combine sequential run directories into one offline experiment report."""

    rounds: list[dict[str, Any]] = []
    test_metrics: dict[str, Any] = {}
    for run_dir in map(Path, run_dirs):
        root = run_dir.expanduser().resolve()
        for checkpoint in sorted((root / "checkpoints").glob("round-*")):
            trajectory = _read_json(checkpoint / "trajectory.json")
            attribution = _read_json(checkpoint / "attribution.json")
            pools = _read_json(checkpoint / "rollout-factor-pools.json")
            if not trajectory:
                continue
            rounds.append(
                {
                    "round": len(rounds) + 1,
                    "source": str(checkpoint),
                    "trajectory": trajectory,
                    "attribution": attribution,
                    "pools": pools,
                }
            )
        candidate = _read_json(root / "backtest-test.json")
        if candidate:
            test_metrics = candidate.get("metrics", {})

    payload = {"title": title, "rounds": rounds, "test_metrics": test_metrics}
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_document(html.escape(title), encoded), encoding="utf-8")
    return target


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _document(title: str, payload: str) -> str:
    template = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--paper:#f7f8f6;--ink:#17202a;--blue:#173b67;--green:#2f6b4f;--orange:#c8643b;--line:#ccd4d9;--soft:#e9eef0;--muted:#65717a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}}
button{{font:inherit}} .page{{max-width:1440px;margin:auto;padding:38px 46px 64px}}
header{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:32px;align-items:end;border-bottom:3px solid var(--blue);padding-bottom:20px}}
h1{{margin:0;font:600 38px/1.1 Georgia,"Times New Roman",serif;color:var(--blue)}}
.subtitle{{max-width:760px;margin:10px 0 0;color:var(--muted)}}
.run-meta{{text-align:right;color:var(--muted)}} .run-meta strong{{display:block;font:600 24px/1.1 Georgia,serif;color:var(--ink)}}
.summary{{display:grid;grid-template-columns:repeat(5,1fr);border-bottom:1px solid var(--line)}}
.metric{{padding:18px 18px 16px;border-right:1px solid var(--line)}} .metric:last-child{{border:0}}
.metric span{{display:block;color:var(--muted);font-size:12px}} .metric b{{font:600 24px/1.25 Georgia,serif}}
h2{{font:600 24px/1.2 Georgia,serif;color:var(--blue);margin:34px 0 16px}}
.lineage{{overflow-x:auto;padding-bottom:8px}} .lineage-track{{display:flex;min-width:max-content;align-items:stretch}}
.round-node{{width:250px;padding:0 18px 18px 0;position:relative}} .round-node:not(:last-child)::after{{content:"";position:absolute;right:3px;top:31px;width:13px;border-top:2px solid var(--blue)}}
.round-label{{display:flex;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:7px;margin-bottom:9px;color:var(--muted)}}
.pool-factor{{padding:6px 9px;margin:5px 0;background:white;border-left:3px solid var(--green);white-space:nowrap}}
.pool-factor.changed{{border-color:var(--orange);background:#fff4ef}}
.round-tabs{{display:flex;gap:4px;border-bottom:1px solid var(--line)}}
.round-tab{{border:0;border-bottom:3px solid transparent;background:none;padding:11px 16px;cursor:pointer;color:var(--muted)}}
.round-tab.active{{color:var(--blue);border-color:var(--blue);font-weight:650}}
.round-view{{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(330px,.8fr);gap:28px;padding-top:22px}}
.step{{display:grid;grid-template-columns:68px minmax(0,1fr);border-top:1px solid var(--line);padding:18px 0}}
.step:first-child{{border-top:0;padding-top:0}} .step-index{{font:600 30px/1 Georgia,serif;color:#9aa8b1}}
.step-head{{display:flex;gap:10px;align-items:center;flex-wrap:wrap}} .step-head h3{{margin:0;font-size:16px}}
.pill{{padding:2px 8px;border-radius:20px;background:var(--soft);font-size:12px}} .accepted{{color:var(--green)}} .rejected{{color:var(--orange)}}
.equation{{font-family:"SFMono-Regular",Consolas,monospace;background:white;border:1px solid var(--line);padding:10px 12px;margin:10px 0;overflow:auto}}
.numbers{{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted)}} .numbers b{{color:var(--ink)}}
.context{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}} .context div{{border-left:2px solid var(--line);padding-left:10px}}
details{{margin-top:11px}} summary{{cursor:pointer;color:var(--blue)}} .message{{border-top:1px solid var(--line);padding:9px 0;white-space:pre-wrap;overflow-wrap:anywhere}}
.message-role{{font-weight:650;color:var(--muted)}} pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 "SFMono-Regular",Consolas,monospace}}
.side-section{{border-top:3px solid var(--green);padding-top:12px;margin-bottom:28px}} .side-section h3{{margin:0 0 10px;font:600 18px Georgia,serif}}
.evidence{{padding:9px 0;border-top:1px solid var(--line)}} .credit-pos{{color:var(--green)}} .credit-neg{{color:var(--orange)}}
.proposal{{background:white;border:1px solid var(--line);padding:12px;margin:9px 0}} .proposal p{{margin:7px 0}}
.diff{{display:grid;grid-template-columns:1fr 1fr;gap:8px}} .diff pre{{margin:0;padding:10px;background:white;border:1px solid var(--line)}} .diff .after{{border-color:#91b39f;background:#f2f8f4}}
.empty{{color:var(--muted);font-style:italic}} .test{{border-left:4px solid var(--orange);padding-left:16px}}
@media(max-width:900px){{.page{{padding:24px 18px}}header{{grid-template-columns:1fr}}.run-meta{{text-align:left}}.summary{{grid-template-columns:repeat(2,1fr)}}.round-view{{grid-template-columns:1fr}}.context,.diff{{grid-template-columns:1fr}}}}
@media print{{.round-tabs{{display:none}}.round-view{{display:block}}details>*{{display:block}}}}
</style></head><body><main class="page">
<header><div><h1>{title}</h1><p class="subtitle">从环境动作到语义归因：因子池演化、回报、组件影响与 harness 修改的完整可追溯记录。</p></div><div class="run-meta"><strong id="round-count">—</strong>连续训练轮次</div></header>
<section class="summary" id="summary"></section>
<h2>因子池谱系</h2><section class="lineage"><div class="lineage-track" id="lineage"></div></section>
<h2>逐轮记录</h2><nav class="round-tabs" id="tabs"></nav><section id="round"></section>
<h2>独立测试集</h2><section class="test" id="test"></section>
</main><script id="data" type="application/json">{payload}</script><script>
const D=JSON.parse(document.querySelector('#data').textContent), rounds=D.rounds;
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const num=(x,n=4)=>x==null?'—':Number(x).toFixed(n); const pct=x=>x==null?'—':(100*Number(x)).toFixed(2)+'%';
const steps=rounds.flatMap(r=>r.trajectory.steps||[]), attrs=rounds.flatMap(r=>r.attribution.attributions||[]), edits=rounds.flatMap(r=>r.attribution.edits||[]);
const tokens=rounds.reduce((a,r)=>a+(r.trajectory.usage?.total_tokens||0),0), reward=steps.reduce((a,s)=>a+(s.reward||0),0);
document.querySelector('#round-count').textContent=rounds.length;
document.querySelector('#summary').innerHTML=[['状态转移',steps.length],['被接受',steps.filter(s=>s.accepted).length],['累计 reward',num(reward,5)],['Harness 修改',edits.length],['Token',tokens.toLocaleString()]].map(x=>`<div class="metric"><span>${x[0]}</span><b>${x[1]}</b></div>`).join('');
let previous=[]; document.querySelector('#lineage').innerHTML=rounds.map(r=>{{let pool=r.pools.final_factors||[], old=new Set(previous.map(x=>x.name));let out=`<article class="round-node"><div class="round-label"><b>Round ${r.round}</b><span>${pool.length} factors</span></div>`+pool.map(f=>`<div class="pool-factor ${{old.has(f.name)?'':'changed'}}" title="${{esc(f.expression)}}">${{esc(f.name)}}</div>`).join('')+'</article>';previous=pool;return out}}).join('');
document.querySelector('#tabs').innerHTML=rounds.map((r,i)=>`<button class="round-tab ${{i===0?'active':''}}" data-i="${{i}}">Round ${{r.round}}</button>`).join('');
document.querySelector('#tabs').onclick=e=>{{if(!e.target.dataset.i)return;document.querySelectorAll('.round-tab').forEach(x=>x.classList.remove('active'));e.target.classList.add('active');renderRound(+e.target.dataset.i)}};
function actionText(a){{if(!a)return 'No action';let f=a.factor;return a.operation==='remove'?`remove(${{a.target}})`:a.operation==='replace'?`replace(${{a.target}}, ${{f?.name}})`:`add(${{f?.name}})`}}
function messages(xs){{return (xs||[]).map(m=>`<div class="message"><span class="message-role">${{esc(m.role)}}${{m.tool_name?' · '+esc(m.tool_name):''}}</span>${{m.content?`<div>${{esc(m.content)}}</div>`:''}}${{m.tool_calls?.length?`<pre>${{esc(JSON.stringify(m.tool_calls,null,2))}}</pre>`:''}}${{m.tool_output!=null?`<pre>${{esc(JSON.stringify(m.tool_output,null,2))}}</pre>`:''}}${{m.tool_error?`<div class="rejected">${{esc(m.tool_error)}}</div>`:''}}</div>`).join('')}}
function renderRound(i){{let r=rounds[i], a=r.attribution||{{}}, ss=r.trajectory.steps||[];document.querySelector('#round').innerHTML=`<div class="round-view"><div><div class="numbers"><span>stop <b>${{esc(r.trajectory.stop_reason)}}</b></span><span>model steps <b>${{r.trajectory.model_steps}}</b></span><span>tokens <b>${{(r.trajectory.usage?.total_tokens||0).toLocaleString()}}</b></span></div>${{ss.map(s=>`<article class="step"><div class="step-index">${{s.step_index+1}}</div><div><div class="step-head"><h3>${{esc(actionText(s.action))}}</h3><span class="pill ${{s.accepted?'accepted':'rejected'}}">${{s.accepted?'accepted':'rejected'}}</span></div>${{s.action?.factor?`<div class="equation">${{esc(s.action.factor.expression)}}</div>`:''}}<div class="numbers"><span>reward <b>${{num(s.reward,6)}}</b></span><span>return <b>${{num(s.return_to_go,6)}}</b></span><span>baseline <b>${{num(s.baseline,6)}}</b></span><span>advantage <b>${{num(s.advantage,6)}}</b></span><span>redundancy <b>${{num(s.redundancy,4)}}</b></span></div><div class="context"><div><b>Skill</b><br>${{esc(s.skill?.id)}}<br><span class="empty">${{esc(s.skill?.description)}}</span></div><div><b>Memory query</b><br>${{esc(s.memory_query)}}<br><span class="empty">${{(s.memory||[]).map(x=>esc(x.id)).join(', ')||'no retrieval'}}</span></div></div><details><summary>查看完整 agent interaction</summary>${{messages(s.interaction)}}</details></div></article>`).join('')||'<p class="empty">本轮没有环境转移。</p>'}}</div><aside><section class="side-section"><h3>组件归因</h3>${{(a.attributions||[]).map(x=>`<div class="evidence"><b>${{esc(x.component)}} · ${{esc(x.item_id)}}</b><br><span>step ${{x.step_index+1}} · influence ${{num(x.influence,5)}} · </span><span class="${{x.credit>=0?'credit-pos':'credit-neg'}}">credit ${{num(x.credit,7)}}</span></div>`).join('')||'<p class="empty">没有归因记录。</p>'}}</section><section class="side-section"><h3>语义修改提案</h3>${{(a.proposals||[]).map(x=>`<div class="proposal"><b>${{esc(x.parameter)}} · score ${{num(x.score,7)}}</b><p>${{esc(x.feedback)}}</p><small>support: ${{esc((x.support_ids||[]).join(', '))}}</small></div>`).join('')||'<p class="empty">没有有效提案。</p>'}}</section><section class="side-section"><h3>实际修改</h3>${{(a.edits||[]).map(x=>`<p><b>${{esc(x.component)}} · ${{esc(x.item_id)}}</b></p><div class="diff"><pre>${{esc(x.before)}}</pre><pre class="after">${{esc(x.after)}}</pre></div>`).join('')||'<p class="empty">本轮未修改 harness。</p>'}}</section></aside></div>`}}
let m=D.test_metrics||{{}};document.querySelector('#test').innerHTML=Object.keys(m).length?`<div class="numbers"><span>IC <b>${{num(m.ic,5)}}</b></span><span>Rank IC <b>${{num(m.rank_ic,5)}}</b></span><span>ICIR <b>${{num(m.icir,5)}}</b></span><span>毛收益 <b>${{pct(m.gross_total_return)}}</b></span><span>扣费收益 <b>${{pct(m.total_return)}}</b></span><span>换手率 <b>${{pct(m.turnover)}}</b></span><span>最大回撤 <b>${{pct(m.max_drawdown)}}</b></span></div>`:'<p class="empty">未找到测试集回测记录。</p>';
if(rounds.length)renderRound(0);
</script></body></html>'''
    template = template.replace("{{", "{").replace("}}", "}")
    return template.replace("{title}", title).replace("{payload}", payload)
