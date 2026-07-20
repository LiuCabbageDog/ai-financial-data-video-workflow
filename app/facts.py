"""Normalize provider quantities and compile fact references into display artifacts."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from math import ceil, floor
from typing import Any

from .models import CanonicalFact, PlanningBundle, ProviderPlanningBundle

MAGNITUDES={"ones":Decimal("1"),"thousands":Decimal("1000"),"millions":Decimal("1000000"),"billions":Decimal("1000000000"),"trillions":Decimal("1000000000000")}
COMPACT=[(Decimal("1000000000000"),"T","trillions"),(Decimal("1000000000"),"B","billions"),(Decimal("1000000"),"M","millions"),(Decimal("1000"),"K","thousands")]


def _decimal(value:float|int) -> Decimal:
    return Decimal(str(value))


def _number(value:Decimal,precision:int) -> str:
    quantum=Decimal("1").scaleb(-precision)
    rendered=f"{value.quantize(quantum,rounding=ROUND_HALF_UP):.{precision}f}"
    return rendered


def normalize_provider_fact(raw:dict[str,Any]) -> dict[str,Any]:
    """Produce one base-unit canonical fact while retaining source presentation."""
    quantity=dict(raw["quantity"]); kind=quantity["kind"]
    if kind in {"money","money_per_share"}:
        base=_decimal(quantity["amount"])*MAGNITUDES[quantity["magnitude"]]
        canonical_quantity={"kind":kind,"amount":float(base),"currency":quantity["currency"]}
        value=float(base); unit="USD" if kind=="money" else "USD per share"; currency=quantity["currency"]
    elif kind=="count":
        base=_decimal(quantity["value"])*MAGNITUDES[quantity["magnitude"]]
        canonical_quantity={"kind":"count","value":float(base),"subject":quantity["subject"]}
        value=float(base); unit=quantity["subject"]; currency=None
    else:
        canonical_quantity=quantity; value=float(quantity["value"]); currency=None
        unit={"percentage":"percent","percentage_points":"percentage_points","ratio":"ratio"}[kind]
    return {
        "id":raw["id"],"metric":raw["metric"],"value":value,"unit":unit,"scale":"ones","currency":currency,
        "basis":raw["basis"],"fiscal_period":raw["fiscal_period"],"period_end":raw["period_end"],"comparison":{},
        "source":raw["source"],"source_locator":raw["source_locator"],"confidence":raw["confidence"],
        "derived_from":raw["derived_from"],"formula":raw["formula"],"quantity":canonical_quantity,
        "reported":{"value":raw["reported_value"],"unit_text":raw["reported_unit_text"]},
    }


def format_quantity(fact:dict[str,Any],precision:int,compact:bool) -> str:
    """Format a canonical quantity; numeric text is never authored by the model."""
    quantity=fact.get("quantity") or _legacy_quantity(fact); kind=quantity["kind"]
    if kind in {"money","money_per_share"}:
        value=_decimal(quantity["amount"]); suffix=""
        if compact:
            for divisor,abbreviation,_ in COMPACT:
                if abs(value)>=divisor: value/=divisor; suffix=abbreviation; break
        rendered=f"${_number(value,precision)}{suffix}"
        return rendered+("/股" if kind=="money_per_share" else "")
    if kind in {"percentage","percentage_points"}:
        rendered=_number(_decimal(quantity["value"]),precision)
        return rendered+("%" if kind=="percentage" else "个百分点")
    value=_decimal(quantity["value"]); suffix=""
    if compact and kind=="count":
        for divisor,abbreviation,_ in COMPACT:
            if abs(value)>=divisor: value/=divisor; suffix=abbreviation; break
    return f"{_number(value,precision)}{suffix}"


def _legacy_quantity(fact:dict[str,Any]) -> dict[str,Any]:
    """Support deterministic fixtures while production migrates to Quantity."""
    unit=str(fact.get("unit","")).lower()
    if fact.get("currency"):
        kind="money_per_share" if "share" in unit else "money"
        return {"kind":kind,"amount":fact["value"],"currency":fact["currency"]}
    if unit in {"percent","%"}: return {"kind":"percentage","value":fact["value"],"semantics":"level"}
    return {"kind":"count","value":fact["value"],"subject":fact.get("unit") or "count"}


def _chart_scale(facts:list[dict[str,Any]],compact:bool) -> tuple[Decimal,str]:
    if not compact: return Decimal("1"),facts[0]["unit"] if facts else ""
    quantities=[f.get("quantity") or _legacy_quantity(f) for f in facts]
    if quantities and all(q["kind"]=="money" for q in quantities):
        maximum=max(abs(_decimal(q["amount"])) for q in quantities)
        for divisor,_,name in COMPACT:
            if maximum>=divisor: return divisor,f"USD {name}"
    if quantities and all(q["kind"]=="count" for q in quantities):
        maximum=max(abs(_decimal(q["value"])) for q in quantities)
        for divisor,_,name in COMPACT:
            if maximum>=divisor: return divisor,name
    return Decimal("1"),facts[0]["unit"] if facts else ""


def compile_provider_bundle(bundle:ProviderPlanningBundle) -> PlanningBundle:
    """Normalize and validate the complete provider bundle before any review gate."""
    raw=bundle.model_dump(mode="json"); facts_raw=raw["canonical_facts"]
    normalized=[normalize_provider_fact(fact) for fact in facts_raw["facts"]]
    fact_by_id={fact["id"]:fact for fact in normalized}
    if len(fact_by_id)!=len(normalized): raise ValueError("provider returned duplicate canonical fact IDs")
    for fact in normalized:
        missing=set(fact["derived_from"])-set(fact_by_id)
        if missing: raise ValueError(f"fact {fact['id']} derives from unknown facts: {sorted(missing)}")
    for section in (raw["financial_analysis"]["insights"],raw["story_plan"]["beats"]):
        for item in section:
            missing=set(item["fact_ids"])-set(fact_by_id)
            if missing: raise ValueError(f"planning item references unknown facts: {sorted(missing)}")

    narration_segments=[]
    for segment in raw["narration"]["segments"]:
        display=[]; fact_ids=[]
        for part in segment["parts"]:
            if part["type"]=="text": display.append(part["value"]); continue
            fact=fact_by_id.get(part["fact_id"])
            if not fact: raise ValueError(f"narration references unknown fact: {part['fact_id']}")
            fact_ids.append(part["fact_id"]); display.append(format_quantity(fact,part["precision"],part["compact"]))
        text="".join(display).strip()
        if not text: raise ValueError(f"narration scene {segment['scene_id']} is empty")
        narration_segments.append({"scene_id":segment["scene_id"],"display_text":text,"spoken_text":text,"fact_ids":list(dict.fromkeys(fact_ids))})

    charts=[]
    for chart in raw["chart_spec"]["charts"]:
        chart_facts=[]
        for point in chart["series"]:
            fact=fact_by_id.get(point["fact_id"])
            if not fact: raise ValueError(f"chart {chart['id']} references unknown fact: {point['fact_id']}")
            chart_facts.append(fact)
        divisor,unit=_chart_scale(chart_facts,chart["compact"]); values=[]; formatted_values=[]; units=[]; key_levels=[]; role_values={}
        for point,fact in zip(chart["series"],chart_facts):
            quantity=fact.get("quantity") or _legacy_quantity(fact)
            raw_value=quantity.get("amount",quantity.get("value")); scaled=_decimal(raw_value)/divisor
            value=float(_number(scaled,chart["precision"]))
            values.append(value); role_values[point["role"]]=value
            formatted_values.append(format_quantity(fact,chart["precision"],chart["compact"]))
            units.append(fact["unit"])
            if point["role"]!="value": key_levels.append({"kind":point["role"],"value":value})
        distinct_units=set(units)
        chart_type="table" if len(distinct_units)>1 and chart["type"] not in {"metric_cards","table"} else chart["type"]
        charts.append({"id":chart["id"],"type":chart_type,"title":chart["title"],"labels":[p["label"] for p in chart["series"]],"values":values,"formatted_values":formatted_values,"units":units,"unit":unit if len(distinct_units)==1 else None,"midpoint":role_values.get("midpoint"),"low":role_values.get("low"),"high":role_values.get("high"),"fact_ids":[p["fact_id"] for p in chart["series"]],"key_levels":key_levels,"animation":chart["animation"]})

    scenes=raw["scene_plan"]; chart_by_id={chart["id"]:chart for chart in charts}; chart_ids=set(chart_by_id)
    for scene in scenes:
        selected=chart_by_id.get(scene.get("chart"))
        if not selected: continue
        meaning=f"{scene.get('purpose','')} {scene.get('title','')} {selected.get('title','')}".lower()
        is_composition=any(token in meaning for token in ("share","mix","contribution","proportion","占比","构成","贡献","占主导"))
        is_growth_comparison=any(token in meaning for token in ("growth rate","growth rates","同比增长","增速对比","增长率"))
        if is_composition and selected["type"]=="bar" and len(selected["values"])==2 and len(set(selected["units"]))==1:
            total_index=0 if selected["values"][0]>=selected["values"][1] else 1; part_index=1-total_index
            total=selected["values"][total_index]; part=selected["values"][part_index]
            selected["type"]="donut"; selected["values"]=[part,max(0,total-part)]; selected["labels"]=[selected["labels"][part_index],"其他"]
            selected["formatted_values"]=[]
        elif is_growth_comparison and selected["type"]=="bar" and len(selected["values"])>=2 and len(set(selected["units"]))==1:
            selected["type"]="line"
        elif selected["type"]=="bar" and len(selected["values"])>=4 and len(set(selected["units"]))==1:
            selected["type"]="horizontal_bar"
    # The renderer template is the authoritative visual contract. Provider models
    # occasionally label a metric-card chart as a generic chart (or vice versa),
    # so reconcile that redundant label before applying aggregate mix rules.
    for scene in scenes:
        if scene["kind"]=="disclaimer":
            scene["visual_kind"]="disclaimer"; scene["chart"]=None
        elif scene.get("chart") in chart_by_id:
            scene["visual_kind"]="metric_cards" if chart_by_id[scene["chart"]]["type"]=="metric_cards" else "chart"
    scene_ids={scene["id"] for scene in scenes}
    disclaimers=[scene for scene in scenes if scene["kind"]=="disclaimer"]
    if len(disclaimers)!=1: raise ValueError("scene plan must contain exactly one disclaimer scene")
    if scenes[-1]["kind"]!="disclaimer": raise ValueError("the disclaimer scene must be last")
    if scenes[-1]["duration_seconds"]<4: raise ValueError("the disclaimer scene must last at least 4 seconds")
    content=[scene for scene in scenes if scene["kind"]!="disclaimer"]
    data_scenes=[scene for scene in content if scene["visual_kind"] in {"chart","metric_cards"}]
    broll=[scene for scene in content if scene["visual_kind"]=="broll"]
    if len(data_scenes)<ceil(len(content)*.8): raise ValueError("at least 80% of non-disclaimer scenes must use chart or metric_cards visuals")
    if len(broll)>floor(len(content)*.2): raise ValueError("broll may occupy at most 20% of non-disclaimer scenes")
    if sum(scene["visual_kind"]=="chart" for scene in content)<2: raise ValueError("scene plan must contain at least two chart scenes")
    if not any(scene["visual_kind"]=="metric_cards" for scene in content): raise ValueError("scene plan must contain at least one metric_cards scene")
    narration_ids={segment["scene_id"] for segment in narration_segments}
    missing_narration=scene_ids-narration_ids
    if missing_narration: raise ValueError(f"scenes missing narration: {sorted(missing_narration)}")
    for scene in scenes:
        if scene["chart"] and scene["chart"] not in chart_ids: raise ValueError(f"scene references unknown chart: {scene['chart']}")
        if scene["kind"]=="disclaimer" and scene["visual_kind"]!="disclaimer": raise ValueError("disclaimer scene must use visual_kind=disclaimer")
        if scene["kind"]!="disclaimer" and scene["visual_kind"]=="disclaimer": raise ValueError("content scene cannot use visual_kind=disclaimer")
        if scene["visual_kind"] in {"chart","metric_cards"} and not scene["chart"]: raise ValueError(f"data visual scene {scene['id']} must reference a chart")
        if scene["visual_kind"] in {"broll","disclaimer"} and scene["chart"]: raise ValueError(f"scene {scene['id']} cannot combine {scene['visual_kind']} with a chart")
        if scene["chart"]:
            selected=chart_by_id[scene["chart"]]
            if (scene["visual_kind"]=="metric_cards") != (selected["type"]=="metric_cards"):
                raise ValueError(f"scene {scene['id']} visual_kind does not match chart type")

    canonical={**facts_raw,"facts":normalized}
    caption=raw["chart_spec"]["caption_region"]
    return PlanningBundle.model_validate({"canonical_facts":canonical,"financial_analysis":raw["financial_analysis"],"story_plan":raw["story_plan"],"scene_plan":{"scenes":scenes},"narration":{"language":raw["narration"]["language"],"segments":narration_segments,"source":raw["narration"]["source"],"editing_applied":raw["narration"]["editing_applied"]},"chart_spec":{"reserved_regions":{"captions":caption},"charts":charts}})
