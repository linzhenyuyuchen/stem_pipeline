#!/usr/bin/env python3
"""Build the self-contained 92B/505B vs Qwen MMMU-Pro comparison page."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).with_name("pangu_vs_qwen_sampling_offline.html")
OLD_DATA = Path(__file__).with_name("comparison_data.json")
LABELS = ROOT / "portrait/labels.parquet"
QWEN = ROOT / "results/qwen3.8-27b/MMMU-Pro/qwen3.8-27b_MMMU-Pro_result.json"
STAGE2_MODEL = ROOT / "results/pangu/MMMU-Pro-COT/92BVL_stage2_part3_091_iter_0008000_MMMU-Pro-COT_result.json"
STAGE2_MAPPING = Path(__file__).with_name("case_comparison.csv")
MODEL_FILES = {
    "92B": ROOT / "results/mid-train/92B/MMMU-Pro-COT/92BVL_32K_midtrain_92B_PILOT_0922_iter0004000_MMMU-Pro-COT_result_fix.json",
    "505B": ROOT / "results/mid-train/505B/MMMU-Pro-COT/92BVL_32K_midtrain_505B_PILOT_0922_iter0004000_MMMU-Pro-COT_result_fix.json",
}
MIDTRAIN_HTML = ROOT / "stage2_data_html/qa_overall_mid_train.html"


def result_rows(path: Path) -> list[dict]:
    return json.loads(path.read_text())["result_details"]


def extract_const(path: Path, name: str) -> dict:
    text = path.read_text(errors="replace")
    marker = f"const {name}="
    start = text.index(marker) + len(marker)
    return json.JSONDecoder().raw_decode(text[start:])[0]


def merge_metric(left: dict, right: dict) -> dict:
    counts = Counter(left.get("counts", {}))
    counts.update(right.get("counts", {}))
    trunc = [x for x in (left.get("truncated_to"), right.get("truncated_to")) if x]
    retained = [x for x in (left.get("min_retained_count"), right.get("min_retained_count")) if x]
    return {
        "counts": dict(counts),
        "total": left.get("total", 0) + right.get("total", 0),
        "rows_with_value": left.get("rows_with_value", 0) + right.get("rows_with_value", 0),
        "occurrences": left.get("occurrences", 0) + right.get("occurrences", 0),
        "truncated_to": sum(trunc) if trunc else None,
        "min_retained_count": min(retained) if retained else None,
    }


def normalize_training_family(source: dict, files: list[str]) -> dict:
    family = {"rows": source["rows"], "files": files, "dimensions": {}}
    for dim in ("domain", "genre", "knowledge_points"):
        metric = dict(source["dimensions"][dim])
        metric["counts"] = dict(metric.get("counts", {}))
        if dim in ("domain", "genre"):
            normalized = Counter()
            for key, count in metric["counts"].items():
                normalized[key.replace(" / ", "-")] += count
            metric["counts"] = dict(normalized)
        family["dimensions"][dim] = metric
    return family


def training_data() -> dict:
    old = json.loads(OLD_DATA.read_text())["training_distribution"]
    caption = normalize_training_family(old["cap"], ["cap_part1.html", "cap_part2.html", "cap_part3.html"])
    qa = normalize_training_family(old["qa"], ["qa_part1.html", "qa_part2.html"])

    bundle = extract_const(MIDTRAIN_HTML, "B")
    metadata = bundle["agg"]["all"]["metadata"]
    mid = {"rows": bundle["agg"]["all"]["rows"], "files": [MIDTRAIN_HTML.name], "dimensions": {}}
    for dim, field in (("domain", "domain_label"), ("genre", "genre_label"), ("knowledge_points", "knowledge_points")):
        src = metadata[field]
        mid["dimensions"][dim] = {
            "counts": src["counts"],
            "total": src.get("count_total", src.get("occurrences", 0)),
            "rows_with_value": src.get("rows_with_value", 0),
            "occurrences": src.get("occurrences", 0),
            "truncated_to": src.get("truncated_to"),
            "min_retained_count": None,
        }
    return {"stage2_caption": caption, "stage2_qa": qa, "midtrain": mid}


def clean_list(value) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    return list(dict.fromkeys(str(x).strip() for x in value if str(x).strip() and str(x) != "nan"))


def label_value(row, prefix: str) -> str:
    code = row.get(f"{prefix}_top1")
    name = row.get(f"{prefix}_top1_name")
    if pd.isna(code) or pd.isna(name):
        return ""
    return f"{name} [{code}]"


def aggregate(case_rows: list[dict], dim: str) -> list[dict]:
    groups: dict[str, dict] = {}
    for row in case_rows:
        vals = row[dim] if isinstance(row[dim], list) else ([row[dim]] if row[dim] else [])
        for value in vals:
            g = groups.setdefault(value, {"category": value, "n": 0, "model_correct": 0, "qwen_correct": 0})
            g["n"] += 1
            g["model_correct"] += int(row["model_correct"])
            g["qwen_correct"] += int(row["qwen_correct"])
    out = []
    for g in groups.values():
        n = g["n"]
        g["model_accuracy"] = g["model_correct"] / n
        g["qwen_accuracy"] = g["qwen_correct"] / n
        g["delta_pp"] = (g["model_accuracy"] - g["qwen_accuracy"]) * 100
        out.append(g)
    return sorted(out, key=lambda x: (-x["n"], x["category"]))


def aggregate_stage2(case_rows: list[dict], dim: str) -> list[dict]:
    groups: dict[str, dict] = {}
    for row in case_rows:
        vals = row[dim] if isinstance(row[dim], list) else ([row[dim]] if row[dim] else [])
        for value in vals:
            g = groups.setdefault(value, {"category": value, "n": 0, "midtrain_correct": 0, "stage2_correct": 0})
            g["n"] += 1
            g["midtrain_correct"] += int(row["midtrain_correct"])
            g["stage2_correct"] += int(row["stage2_correct"])
    out = []
    for g in groups.values():
        n = g["n"]
        g["midtrain_accuracy"] = g["midtrain_correct"] / n
        g["stage2_accuracy"] = g["stage2_correct"] / n
        g["uplift_pp"] = (g["midtrain_accuracy"] - g["stage2_accuracy"]) * 100
        out.append(g)
    return sorted(out, key=lambda x: (-x["n"], x["category"]))


def comparison_data() -> dict:
    labels = pd.read_parquet(LABELS).reset_index(drop=True)
    qwen = result_rows(QWEN)
    if len(labels) != len(qwen):
        raise ValueError(f"labels ({len(labels)}) and Qwen ({len(qwen)}) length mismatch")
    models = {}
    for model_name, model_path in MODEL_FILES.items():
        model = result_rows(model_path)
        if len(model) != len(labels):
            raise ValueError(f"{model_name} ({len(model)}) and labels ({len(labels)}) length mismatch")
        cases = []
        for i, lab in labels.iterrows():
            cases.append({
                "id": int(model[i].get("id", i)),
                "domain": label_value(lab, "domain"),
                "genre": label_value(lab, "genre"),
                "knowledge_points": clean_list(lab.get("knowledge_points")),
                "model_correct": bool(model[i].get("eval_result")),
                "qwen_correct": bool(qwen[i].get("eval_result")),
            })
        mc = sum(x["model_correct"] for x in cases)
        qc = sum(x["qwen_correct"] for x in cases)
        n = len(cases)
        models[model_name] = {
            "overall": {"n": n, "model_correct": mc, "qwen_correct": qc, "model_accuracy": mc / n, "qwen_accuracy": qc / n, "delta_pp": (mc - qc) / n * 100},
            "dimensions": {dim: aggregate(cases, dim) for dim in ("domain", "genre", "knowledge_points")},
            "case_rows": cases,
        }
    stage2_rows = result_rows(STAGE2_MODEL)
    mapping = pd.read_csv(STAGE2_MAPPING)
    paired = []
    for rec in mapping.to_dict("records"):
        qid, pid = int(rec["qwen_result_id"]), int(rec["pangu_result_id"])
        current = models["92B"]["case_rows"][qid]
        paired.append({
            "index": rec["index"], "domain": current["domain"], "genre": current["genre"],
            "knowledge_points": current["knowledge_points"],
            "midtrain_correct": current["model_correct"],
            "stage2_correct": bool(stage2_rows[pid].get("eval_result")),
        })
    mid_correct = sum(x["midtrain_correct"] for x in paired)
    stage2_correct = sum(x["stage2_correct"] for x in paired)
    paired_comparison = {
        "overall": {
            "n": len(paired), "midtrain_correct": mid_correct, "stage2_correct": stage2_correct,
            "midtrain_accuracy": mid_correct / len(paired), "stage2_accuracy": stage2_correct / len(paired),
            "uplift_pp": (mid_correct - stage2_correct) / len(paired) * 100,
        },
        "dimensions": {dim: aggregate_stage2(paired, dim) for dim in ("domain", "genre", "knowledge_points")},
        "case_rows": paired,
    }
    return {
        "models": models,
        "training": training_data(),
        "stage2_comparison": paired_comparison,
        "meta": {
            "labels": str(LABELS), "qwen": str(QWEN), "model_files": {k: str(v) for k, v in MODEL_FILES.items()},
            "stage2_model": str(STAGE2_MODEL), "stage2_mapping": str(STAGE2_MAPPING),
            "alignment": "All 3,460 rows are aligned positionally with labels.parquet, as specified.",
        },
    }


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>92B / 505B vs Qwen | MMMU-Pro 能力差异</title>
<style>
:root{--ink:#263342;--muted:#68788b;--line:#d9e1eb;--blue:#597db4;--red:#c24c47;--green:#39735f;--bg:#f7f9fc;--card:#fff}*{box-sizing:border-box}body{margin:0;background:var(--bg);font-family:Inter,system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:var(--ink);font-size:14px}.wrap{max-width:1500px;margin:auto;padding:26px 28px 48px}.eyebrow{font-size:12px;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}.title{font-size:28px;font-weight:750;margin:5px 0 7px}.sub{color:var(--muted);max-width:1100px;line-height:1.55}.cards{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:22px 0 18px}.card,.panel,.toolbar{background:var(--card);border:1px solid var(--line);border-radius:10px;box-shadow:0 1px 2px #10233d0a}.card{padding:16px 18px}.k{color:var(--muted);font-size:12px}.v{font-size:25px;font-weight:750;margin-top:5px}.neg{color:var(--red)}.pos{color:var(--green)}.toolbar{display:flex;gap:12px;align-items:end;flex-wrap:wrap;padding:14px 16px;margin-bottom:14px}.control label{display:block;color:var(--muted);font-size:12px;margin-bottom:5px}.control select,.control input{border:1px solid #cdd7e3;border-radius:6px;padding:8px 10px;background:white;color:var(--ink)}.hint{margin-left:auto;color:var(--muted);font-size:12px;max-width:530px;line-height:1.4}.panel{overflow:hidden;margin-top:16px}.panel h2{font-size:16px;margin:0;padding:16px 18px 5px}.desc{margin:0;padding:0 18px 14px;color:var(--muted);font-size:12px;line-height:1.5}.tablewrap{overflow:auto;max-height:700px;border-top:1px solid var(--line)}table{width:100%;border-collapse:collapse;min-width:850px}th,td{padding:9px 10px;border-bottom:1px solid #e6ebf1;white-space:nowrap}th{text-align:left;color:var(--muted);background:#f4f7fb;font-size:12px;position:sticky;top:0;z-index:1;cursor:pointer}th:hover{background:#eaf0f8}th[data-sort-direction=asc]::after{content:" ▲";color:var(--blue)}th[data-sort-direction=desc]::after{content:" ▼";color:var(--blue)}td.num{text-align:right;font-variant-numeric:tabular-nums}.cat{max-width:430px;overflow:hidden;text-overflow:ellipsis}.barcell{min-width:150px}.bar{height:14px;background:#eef2f7;border-radius:2px;overflow:hidden}.bar i{display:block;height:100%}.small{font-size:12px;color:var(--muted)}.foot{margin:16px 18px;padding:13px 16px;background:#fff8e8;border:1px solid #ead39d;color:#6f5d39;border-radius:8px;font-size:12px;line-height:1.55}.legend{display:flex;gap:16px;padding:12px 18px;color:var(--muted);font-size:12px;border-top:1px solid var(--line)}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;padding:0 18px 18px}.mini{border:1px solid var(--line);border-radius:8px;padding:14px}.mini h3{margin:0 0 8px;font-size:14px}.insight-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;padding:0 18px 18px}.insight-card{border:1px solid var(--line);border-radius:8px;padding:15px 16px;line-height:1.6}.insight-card h3{margin:0 0 7px;font-size:15px}.insight-card p{margin:6px 0}.insight-card.good{border-left:4px solid var(--green)}.insight-card.partial{border-left:4px solid #d29a36}.insight-card.bad{border-left:4px solid var(--red)}.insight-card.caution{border-left:4px solid var(--blue)}.source{margin-top:18px;color:var(--muted);font-size:12px;line-height:1.6;word-break:break-all}@media(max-width:950px){.cards{grid-template-columns:repeat(2,1fr)}.grid,.insight-grid{grid-template-columns:1fr}.hint{margin-left:0}}@media(max-width:560px){.wrap{padding:18px 12px}.cards{grid-template-columns:1fr}.title{font-size:23px}}
</style></head><body><main class="wrap">
<div class="eyebrow">MMMU-Pro · paired evaluation · offline</div><div class="title">92B / 505B 相比 Qwen 的能力差异</div>
<div class="sub">按结果文件与 <code>labels.parquet</code> 的相同行序一一对齐全部 3,460 个 case，正确性采用 <code>eval_result</code>。可切换 92B 或 505B；Δ(pp) = 所选模型准确率 − Qwen 准确率。训练数据分别展示 <strong>stage2-caption</strong>、<strong>stage2-qa</strong> 与新增的 <strong>mid-train</strong>。</div>
<section class="cards" id="cards"></section>
<section class="toolbar"><div class="control"><label>对比模型</label><select id="model"><option value="92B">92B vs Qwen</option><option value="505B">505B vs Qwen</option></select></div><div class="control"><label>第一维度</label><select id="dimension"></select></div><div class="control"><label>第二维度</label><select id="dimension2"><option value="__none__">无（单维度）</option></select></div><div class="control"><label>最小样本数 n</label><input id="minN" type="number" min="1" value="10"></div><div class="control"><label>显示行数</label><select id="limit"><option>20</option><option selected>40</option><option>80</option><option>200</option><option>1000</option></select></div><div class="hint">knowledge_points 为多标签展开统计；交叉分组时，同一 case 可能进入多个知识点组合。训练列在交叉模式下显示两个维度各自的边际分布。</div></section>
<section class="panel"><h2 id="tableTitle"></h2><p class="desc" id="tableDesc"></p><div class="tablewrap"><table><thead><tr><th>标签 / 子组</th><th>Δ(pp)</th><th>n</th><th>Qwen</th><th id="selectedHead">所选模型</th><th>差异方向</th><th id="captionHead">stage2-caption</th><th id="qaHead">stage2-qa</th><th id="midHead">mid-train</th></tr></thead><tbody id="tbody"></tbody></table></div><div class="legend"><span class="neg">■ 所选模型相对落后</span><span class="pos">■ 所选模型相对领先</span><span>点击任意表头可排序</span></div></section>
<section class="panel"><h2>92B-midtrain vs 92B-stage2-part3</h2><p class="desc" id="stage2Desc"></p><div id="stage2Cards" class="cards" style="padding:0 18px;margin:4px 0 16px"></div><div class="tablewrap"><table><thead><tr><th>标签 / 子组</th><th>提升(pp)</th><th>配对 n</th><th>92B-stage2-part3</th><th>92B-midtrain</th><th>stage2-caption</th><th>stage2-qa</th><th>mid-train</th></tr></thead><tbody id="stage2Body"></tbody></table></div></section>
<section class="panel"><h2>mid-train 弱项上采样建议</h2><p class="desc">随模型选择实时更新。只列所选维度中 Δ&lt;0 的弱项；权重综合评测超额错误数（70%）与 mid-train 相对 stage2 的覆盖稀缺度（30%），线性映射到 1–4×，作为方向性建议。</p><div id="recommendations"></div></section>
<section class="panel"><h2>训练数据概览</h2><p class="desc">绝对量为对应标签 occurrence；占比的分母采用源 HTML 的 <code>count_total</code>。stage2 的长尾知识点由 CAP/QA 各自截断结果合并，mid-train 知识点仅保留 Top 2000，因此未命中项不能解释为真实 0。</p><div class="grid" id="coverage"></div></section>
<section class="panel"><h2>数据量、对 Qwen 表现与 mid-train 提升结论</h2><p class="desc">“数据量多”定义为当前总训练覆盖（stage2-caption + stage2-qa + mid-train）在该维度有效评测标签中的前 25%；“符合预期的提升”定义为 mid-train 新增量位于前 25%，且在 1,500 个严格配对 case 上相对 stage2-part3 提升。domain 按训练报告口径归并到前两级。knowledge_points 为降低小样本波动，结论仅使用配对 n≥3，其余维度使用 n≥10。</p><div id="analysis"></div></section>
<section class="panel"><h2>上层规律与行动建议</h2><p class="desc">将具体标签归纳为可执行的数据规律；“符合预期”表示新增数据与能力提升方向一致，不代表已经追平 Qwen。</p><div class="insight-grid">
<div class="insight-card good"><h3>符合预期：结构化 STEM 知识与标准题型</h3><p><strong>物理力学/电磁学、生物学知识、基础医学、有机化学、概率统计</strong>随 mid-train 增量获得稳定提升；试卷练习、坐标函数图、电路图、一般示意图也同步改善。</p><p><strong>规律：</strong>答案边界清晰、规则稳定、训练与评测形态相近的知识问答，继续增加高质量数据仍有正向收益。</p><p><strong>建议：</strong>保持当前配比，优先扩展难度、表达和图像风格，而不是简单复制同质题。</p></div>
<div class="insight-card partial"><h3>部分符合预期：明显提升，但尚未追平 Qwen</h3><p><strong>数学几何、地图/GIS、几何图形、标注结构图</strong>相对 stage2 提升明显，但当前对 Qwen 仍然落后。</p><p><strong>规律：</strong>新增数据解决了基础覆盖不足，但空间关系、多步推理和复杂图形泛化仍是瓶颈；继续堆同分布数据的边际收益可能下降。</p><p><strong>建议：</strong>加入更难的组合图、多视角/多面板、反事实干扰项和显式推理链，并针对错误类型做定向采样。</p></div>
<div class="insight-card bad"><h3>不符合预期：数据量大，但提升很少或饱和</h3><p><strong>无机化学、物理化学、流程与工作流图、勾股定理</strong>具有较大 mid-train 覆盖，但严格配对提升接近 0。</p><p><strong>规律：</strong>问题更可能来自重复度、难度结构、答案噪声或训练目标不匹配，而非单纯数据不足。</p><p><strong>建议：</strong>暂缓等比例扩量；先做去重与泄漏检查、难度分层、答案验证和失败案例聚类，再补充能覆盖真实错误模式的数据。</p></div>
<div class="insight-card caution"><h3>持续短板与证据不足：需要“小而准”的补强</h3><p><strong>光学、机械工程、折线/面积图、柱状图</strong>在数据较多时仍落后 Qwen；<strong>万有引力、条件概率</strong>也呈负差，但知识点评测样本较少。</p><p><strong>规律：</strong>视觉读数、变量关系和跨步骤计算比知识记忆更难，仅靠知识点命中无法保证能力迁移。</p><p><strong>建议：</strong>先扩充独立验证集确认差距，再加入带中间过程校验的高难样本；采样权重按“错误率 × 可信样本量”设置，避免被小样本极值误导。</p></div>
</div><div class="foot"><strong>总体判断：</strong>mid-train 的主要收益来自结构化 STEM QA 和标准化视觉题型；下一阶段不应继续全面增量，而应从“补覆盖”转向“补难度、补推理过程、补失败模式”。优先处理“已经提升但仍落后 Qwen”的方向，其次审计“数据很多但零提升”的方向，知识点长尾则先扩大评测证据再决定训练投入。</div></section>
<div class="foot">口径说明：domain 与 genre 使用画像 Top-1；knowledge_points 使用全部知识点并展开。训练侧 domain 统一到前两级，以匹配训练报告标签。页面完全离线，切换模型不会发起网络请求。</div>
<div class="source" id="source"></div>
</main><script>
const DATA=__DATA__;
const LABELS={domain:'domain',genre:'genre',knowledge_points:'knowledge_points'};
const modelEl=document.getElementById('model'),dim=document.getElementById('dimension'),dim2=document.getElementById('dimension2'),minN=document.getElementById('minN'),limit=document.getElementById('limit');
Object.keys(LABELS).forEach(k=>{dim.add(new Option(LABELS[k],k));dim2.add(new Option(LABELS[k],k))});
const nf=new Intl.NumberFormat('zh-CN'),pct=x=>(x*100).toFixed(1)+'%',pp=x=>(x>=0?'+':'')+x.toFixed(2),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function trainKey(d,value){let t=String(value||'').replace(/\s*\[[^\]]+\]\s*$/,'').trim();if(d==='knowledge_points')return t;let p=(t.includes(' / ')?t.split(' / '):t.split('-')).map(x=>x.trim()).filter(Boolean);p=p.filter((x,i)=>i===0||x!==p[i-1]);if(d==='domain')p=p.slice(0,2);return p.join('-')}
function metric(family,d,value){const field=DATA.training[family].dimensions[d],key=trainKey(d,value),known=Object.prototype.hasOwnProperty.call(field.counts,key),count=field.counts[key]||0,total=field.total||0;return{count,share:total?count/total:0,censored:!known&&!!field.truncated_to,truncated:field.truncated_to}}
function stage2Metric(d,value){const a=metric('stage2_caption',d,value),b=metric('stage2_qa',d,value),total=DATA.training.stage2_caption.dimensions[d].total+DATA.training.stage2_qa.dimensions[d].total;return{count:a.count+b.count,share:total?(a.count+b.count)/total:0,censored:a.censored&&b.censored,truncated:(a.truncated||0)+(b.truncated||0)}}
function cell(family,d1,a,d2,b){const fmt=m=>m.censored?`低频截断（未进 Top ${nf.format(m.truncated)}）`:`${nf.format(m.count)} (${pct(m.share)})`,one=metric(family,d1,a);return d2?`<div>${fmt(one)}</div><div class="small">${fmt(metric(family,d2,b))}</div>`:fmt(one)}
function values(row,d){const v=row[d];return Array.isArray(v)?[...new Set(v.filter(Boolean))]:(v?[v]:[])}
function crossRows(block,d1,d2){const groups=new Map();block.case_rows.forEach(r=>values(r,d1).forEach(a=>values(r,d2).forEach(b=>{const key=`${a} × ${b}`;if(!groups.has(key))groups.set(key,{category:key,group_a:a,group_b:b,n:0,model_correct:0,qwen_correct:0});const g=groups.get(key);g.n++;g.model_correct+=r.model_correct?1:0;g.qwen_correct+=r.qwen_correct?1:0})));return[...groups.values()].map(g=>({...g,model_accuracy:g.model_correct/g.n,qwen_accuracy:g.qwen_correct/g.n,delta_pp:(g.model_correct-g.qwen_correct)/g.n*100}));}
function renderCards(){const name=modelEl.value,o=DATA.models[name].overall;document.getElementById('cards').innerHTML=`<div class="card"><div class="k">对比 case 数</div><div class="v">${nf.format(o.n)}</div></div><div class="card"><div class="k">Qwen 准确率</div><div class="v">${pct(o.qwen_accuracy)}</div></div><div class="card"><div class="k">${name} 准确率</div><div class="v">${pct(o.model_accuracy)}</div></div><div class="card"><div class="k">${name} − Qwen</div><div class="v ${o.delta_pp<0?'neg':'pos'}">${pp(o.delta_pp)} pp</div></div><div class="card"><div class="k">正确数（${name} / Qwen）</div><div class="v">${nf.format(o.model_correct)} / ${nf.format(o.qwen_correct)}</div></div>`}
function currentRows(){const block=DATA.models[modelEl.value],d1=dim.value,d2=dim2.value;return d2==='__none__'?block.dimensions[d1]:crossRows(block,d1,d2)}
function render(){renderCards();const name=modelEl.value,d1=dim.value,d2=dim2.value,isCross=d2!=='__none__',threshold=Math.max(1,+minN.value||1),lim=+limit.value;let all=currentRows().filter(x=>x.n>=threshold).sort((a,b)=>b.n-a.n),rows=all.slice(0,lim),max=Math.max(...rows.map(x=>Math.abs(x.delta_pp)),1);document.getElementById('selectedHead').textContent=name;document.getElementById('captionHead').textContent='stage2-caption'+(isCross?'（边际）':'');document.getElementById('qaHead').textContent='stage2-qa'+(isCross?'（边际）':'');document.getElementById('midHead').textContent='mid-train'+(isCross?'（边际）':'');document.getElementById('tableTitle').textContent=isCross?`${name} vs Qwen · ${LABELS[d1]} × ${LABELS[d2]}`:`${name} vs Qwen · ${LABELS[d1]}`;document.getElementById('tableDesc').textContent=`共 ${all.length} 个分组满足 n≥${threshold}；当前显示 ${Math.min(rows.length,lim)} 行。`;document.getElementById('tbody').innerHTML=rows.map(x=>{const a=x.group_a||x.category,b=x.group_b||'',w=Math.abs(x.delta_pp)/max*100;return`<tr><td class="cat" title="${esc(x.category)}">${esc(x.category)}</td><td class="num ${x.delta_pp<0?'neg':'pos'}">${pp(x.delta_pp)}</td><td class="num">${x.n}</td><td class="num">${pct(x.qwen_accuracy)}</td><td class="num">${pct(x.model_accuracy)}</td><td class="barcell" data-sort-value="${Math.abs(x.delta_pp)}"><div class="bar"><i style="width:${w}%;background:${x.delta_pp<0?'var(--red)':'var(--green)'}"></i></div></td><td class="num">${cell('stage2_caption',d1,a,isCross?d2:null,b)}</td><td class="num">${cell('stage2_qa',d1,a,isCross?d2:null,b)}</td><td class="num">${cell('midtrain',d1,a,isCross?d2:null,b)}</td></tr>`}).join('')||'<tr><td colspan="9">没有满足阈值的分组。</td></tr>';renderRecommendations();renderStage2Comparison();}
function renderRecommendations(){const d=dim.value,threshold=Math.max(1,+minN.value||1),rows=DATA.models[modelEl.value].dimensions[d].filter(x=>x.n>=threshold&&x.delta_pp<0),maxDemand=Math.max(...rows.map(x=>x.n*-x.delta_pp/100),1);rows.forEach(x=>{const s=stage2Metric(d,x.category),m=metric('midtrain',d,x.category),severity=(x.n*-x.delta_pp/100)/maxDemand,scarcity=s.share+m.share?s.share/(s.share+m.share):1;x.weight=1+3*(.7*severity+.3*scarcity);x.s=s;x.m=m});rows.sort((a,b)=>b.weight-a.weight);document.getElementById('recommendations').innerHTML=`<div class="tablewrap" style="max-height:560px"><table><thead><tr><th>弱项</th><th>评测 n</th><th>Δ(pp)</th><th>超额错误估计</th><th>stage2 合计</th><th>mid-train</th><th>建议权重</th></tr></thead><tbody>${rows.map(x=>`<tr><td class="cat" title="${esc(x.category)}">${esc(x.category)}</td><td class="num">${x.n}</td><td class="num neg">${pp(x.delta_pp)}</td><td class="num">${(x.n*-x.delta_pp/100).toFixed(2)}</td><td class="num">${x.s.censored?'低频截断':`${nf.format(x.s.count)} (${pct(x.s.share)})`}</td><td class="num">${x.m.censored?'低频截断':`${nf.format(x.m.count)} (${pct(x.m.share)})`}</td><td class="num pos">${x.weight.toFixed(2)}×</td></tr>`).join('')}</tbody></table></div>`}
function renderStage2Comparison(){const o=DATA.stage2_comparison.overall,d=dim.value,threshold=Math.max(1,+minN.value||1),lim=+limit.value,rows=DATA.stage2_comparison.dimensions[d].filter(x=>x.n>=threshold).sort((a,b)=>b.n-a.n).slice(0,lim);document.getElementById('stage2Desc').textContent=`使用旧页面已验证的 case_comparison.csv 映射，在相同 1,500 个 case 上严格配对；表格跟随“第一维度”、最小样本数与显示行数。`;document.getElementById('stage2Cards').innerHTML=`<div class="card"><div class="k">严格配对 case</div><div class="v">${nf.format(o.n)}</div></div><div class="card"><div class="k">stage2-part3</div><div class="v">${pct(o.stage2_accuracy)}</div></div><div class="card"><div class="k">midtrain</div><div class="v">${pct(o.midtrain_accuracy)}</div></div><div class="card"><div class="k">midtrain 提升</div><div class="v ${o.uplift_pp>=0?'pos':'neg'}">${pp(o.uplift_pp)} pp</div></div>`;document.getElementById('stage2Body').innerHTML=rows.map(x=>`<tr><td class="cat" title="${esc(x.category)}">${esc(x.category)}</td><td class="num ${x.uplift_pp>=0?'pos':'neg'}">${pp(x.uplift_pp)}</td><td class="num">${x.n}</td><td class="num">${pct(x.stage2_accuracy)}</td><td class="num">${pct(x.midtrain_accuracy)}</td><td class="num">${cell('stage2_caption',d,x.category)}</td><td class="num">${cell('stage2_qa',d,x.category)}</td><td class="num">${cell('midtrain',d,x.category)}</td></tr>`).join('')||'<tr><td colspan="8">没有满足阈值的分组。</td></tr>'}
function renderCoverage(){const families=[['stage2_caption','stage2-caption'],['stage2_qa','stage2-qa'],['midtrain','mid-train']];document.getElementById('coverage').innerHTML=Object.keys(LABELS).map(d=>`<div class="mini"><h3>${LABELS[d]}</h3>${families.map(([k,n])=>{const x=DATA.training[k].dimensions[d];return`<div>${n}：<strong>${nf.format(x.total)}</strong> occurrence，${nf.format(Object.keys(x.counts).length)} 个保留标签</div>`}).join('')}<div class="small" style="margin-top:7px">mid-train 知识点 Top 2000；未命中长尾按截断理解。</div></div>`).join('')}
function regroup(rows,d,type){const g=new Map();rows.forEach(x=>{const k=trainKey(d,x.category);if(!g.has(k))g.set(k,type==='main'?{category:k,n:0,model_correct:0,qwen_correct:0}:{category:k,n:0,midtrain_correct:0,stage2_correct:0});const z=g.get(k);z.n+=x.n;if(type==='main'){z.model_correct+=x.model_correct;z.qwen_correct+=x.qwen_correct}else{z.midtrain_correct+=x.midtrain_correct;z.stage2_correct+=x.stage2_correct}});return[...g.values()].map(x=>type==='main'?({...x,delta_pp:(x.model_correct-x.qwen_correct)/x.n*100}):({...x,uplift_pp:(x.midtrain_correct-x.stage2_correct)/x.n*100}))}
function quantile75(nums){const a=[...nums].sort((x,y)=>x-y);return a.length?a[Math.floor((a.length-1)*.75)]:0}
function analysisTable(title,rows,valueName,valueFn,deltaName,deltaFn){return`<div style="padding:0 18px 16px"><h3 style="font-size:14px;margin:12px 0 7px">${title}</h3><div class="tablewrap" style="max-height:360px"><table><thead><tr><th>标签</th><th>${valueName}</th><th>评测 n</th><th>${deltaName}</th></tr></thead><tbody>${rows.slice(0,10).map(x=>`<tr><td class="cat">${esc(x.category)}</td><td class="num">${nf.format(valueFn(x))}</td><td class="num">${x.n}</td><td class="num ${deltaFn(x)>=0?'pos':'neg'}">${pp(deltaFn(x))}</td></tr>`).join('')||'<tr><td colspan="4">无满足条件的标签</td></tr>'}</tbody></table></div></div>`}
function renderAnalysis(){let html='';Object.keys(LABELS).forEach(d=>{const nMin=d==='knowledge_points'?3:10,main=regroup(DATA.models['92B'].dimensions[d],d,'main').filter(x=>x.n>=nMin),paired=regroup(DATA.stage2_comparison.dimensions[d],d,'paired').filter(x=>x.n>=nMin);main.forEach(x=>x.totalCount=['stage2_caption','stage2_qa','midtrain'].reduce((a,f)=>a+metric(f,d,x.category).count,0));paired.forEach(x=>x.midCount=metric('midtrain',d,x.category).count);const totalCut=quantile75(main.map(x=>x.totalCount).filter(x=>x>0)),midCut=quantile75(paired.map(x=>x.midCount).filter(x=>x>0)),high=main.filter(x=>x.totalCount>0&&x.totalCount>=totalCut),good=high.filter(x=>x.delta_pp>=0).sort((a,b)=>b.totalCount-a.totalCount),bad=high.filter(x=>x.delta_pp<0).sort((a,b)=>b.totalCount-a.totalCount),expected=paired.filter(x=>x.midCount>0&&x.midCount>=midCut&&x.uplift_pp>0).sort((a,b)=>b.midCount-a.midCount),limited=paired.filter(x=>x.midCount>0&&x.midCount>=midCut&&x.uplift_pp<=0).sort((a,b)=>b.midCount-a.midCount);html+=`<div class="foot"><strong>${LABELS[d]} 总结：</strong>高数据量标签 ${high.length} 个，其中相对 Qwen 表现不差 ${good.length} 个、仍落后 ${bad.length} 个；mid-train 高新增量标签中符合预期提升 ${expected.length} 个、提升较少或退化 ${limited.length} 个。</div>`+analysisTable(`${LABELS[d]}：数据多且对 Qwen 表现好`,good,'当前总量',x=>x.totalCount,'92B−Qwen (pp)',x=>x.delta_pp)+analysisTable(`${LABELS[d]}：数据多但对 Qwen 仍不理想`,bad,'当前总量',x=>x.totalCount,'92B−Qwen (pp)',x=>x.delta_pp)+analysisTable(`${LABELS[d]}：mid-train 新增量大且符合预期提升`,expected,'mid-train 量',x=>x.midCount,'较 stage2 提升(pp)',x=>x.uplift_pp)+analysisTable(`${LABELS[d]}：mid-train 新增量大但提升较少`,limited,'mid-train 量',x=>x.midCount,'较 stage2 提升(pp)',x=>x.uplift_pp)});document.getElementById('analysis').innerHTML=html}
function sortValue(cell){if(cell.dataset.sortValue!==undefined)return Number(cell.dataset.sortValue);const t=cell.textContent.trim().replace(/,/g,'');if(cell.classList.contains('num')){const m=t.match(/[+-]?\d+(?:\.\d+)?/);return m?Number(m[0]):null}return t.toLocaleLowerCase('zh-CN')}
document.addEventListener('click',e=>{const th=e.target.closest&&e.target.closest('th');if(!th)return;const table=th.closest('table'),body=table&&table.tBodies[0];if(!body)return;const col=[...th.parentElement.children].indexOf(th),same=table.dataset.col==col,dir=same&&table.dataset.dir==='desc'?'asc':same?'desc':th.parentElement.children[col].nextElementSibling?'desc':'asc',rows=[...body.rows];rows.sort((a,b)=>{const x=sortValue(a.children[col]),y=sortValue(b.children[col]);if(x==null)return 1;if(y==null)return-1;const c=typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y),'zh-CN',{numeric:true});return dir==='asc'?c:-c});rows.forEach(r=>body.appendChild(r));table.querySelectorAll('th').forEach(x=>x.removeAttribute('data-sort-direction'));th.dataset.sortDirection=dir;table.dataset.col=col;table.dataset.dir=dir});
document.getElementById('source').innerHTML=`数据来源：<code>${esc(DATA.meta.qwen)}</code><br>92B-midtrain：<code>${esc(DATA.meta.model_files['92B'])}</code><br>92B-stage2-part3：<code>${esc(DATA.meta.stage2_model)}</code><br>505B：<code>${esc(DATA.meta.model_files['505B'])}</code><br>画像：<code>${esc(DATA.meta.labels)}</code><br>严格配对映射：<code>${esc(DATA.meta.stage2_mapping)}</code><br>训练数据：stage2_data_html/{cap_part1,cap_part2,cap_part3,qa_part1,qa_part2}.html 与 <code>qa_overall_mid_train.html</code>`;
[modelEl,dim,dim2,minN,limit].forEach(x=>x.addEventListener('input',render));renderCoverage();renderAnalysis();render();
</script></body></html>'''


def main() -> None:
    data = comparison_data()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(HTML.replace("__DATA__", payload))
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    for name, block in data["models"].items():
        o = block["overall"]
        print(f"{name}: n={o['n']}, model={o['model_correct']}/{o['n']} ({o['model_accuracy']:.4%}), qwen={o['qwen_correct']}/{o['n']} ({o['qwen_accuracy']:.4%}), delta={o['delta_pp']:.3f} pp")


if __name__ == "__main__":
    main()
