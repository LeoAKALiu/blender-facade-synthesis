"""Local browser-facing Web Studio; it delegates every render to the Worker."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .contracts import GenerationBrief, TaskKind
from .packages import BlenderProcRenderer, fingerprint_local_asset
from .runtime import BlenderProcRuntime
from .studio import StudioService


def create_app(*, workspace: Path, renderer: BlenderProcRenderer | None = None) -> FastAPI:
    """Create a local Studio application with no browser-side render authority."""

    studio = StudioService(workspace=workspace, renderer=renderer or BlenderProcRenderer())
    app = FastAPI(title="Blender Facade Synthesis Studio")
    app.state.studio = studio
    app.state.workspace = studio.workspace
    app.mount("/packages", StaticFiles(directory=studio.workspace / "packages"), name="packages")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _WEB_STUDIO_HTML

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return [job.to_dict() for job in studio.list_jobs()]

    @app.post("/api/jobs", status_code=201)
    def create_job(payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            job = studio.create_job(_brief_from_payload(payload))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return job.to_dict()

    @app.post("/api/jobs/{job_id}/confirm")
    def confirm_job(job_id: str, payload: Mapping[str, str]) -> dict[str, Any]:
        return _job_response(lambda: studio.confirm_brief(job_id, confirmed_by=payload.get("confirmed_by", "")))

    @app.post("/api/jobs/run-next")
    def run_next() -> dict[str, Any]:
        return _job_response(studio.run_next)

    @app.post("/api/jobs/{job_id}/review")
    def review_job(job_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        approved = payload.get("approved")
        if not isinstance(approved, bool):
            raise HTTPException(status_code=422, detail="approved must be a JSON boolean")
        return _job_response(
            lambda: studio.record_review(
                job_id,
                reviewer=str(payload.get("reviewer", "")),
                approved=approved,
            )
        )

    @app.post("/api/jobs/{job_id}/publish")
    def publish_job(job_id: str, payload: Mapping[str, str]) -> dict[str, Any]:
        try:
            return studio.publish(job_id, published_by=payload.get("published_by", "")).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        return _job_response(lambda: studio.cancel(job_id))

    @app.post("/api/jobs/{job_id}/resume")
    def resume_job(job_id: str) -> dict[str, Any]:
        return _job_response(lambda: studio.resume(job_id))

    @app.get("/api/preflight")
    def preflight() -> dict[str, Any]:
        try:
            return BlenderProcRuntime().preflight()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


def _job_response(action: Any) -> dict[str, Any]:
    try:
        return action().to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _brief_from_payload(payload: Mapping[str, Any]) -> GenerationBrief:
    value = dict(payload)
    asset_paths = value.pop("asset_paths", [])
    if asset_paths:
        value["asset_paths"] = tuple(str(Path(path).resolve()) for path in asset_paths)
        value["asset_fingerprints"] = tuple(fingerprint_local_asset(Path(path)) for path in asset_paths)
    value["task"] = TaskKind(value["task"])
    value["view_family"] = tuple(value.get("view_family", ("frontal", "light_medium_oblique", "strong_oblique")))
    return GenerationBrief(**value)


_WEB_STUDIO_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blender Facade Synthesis Studio</title><style>
body{font:15px system-ui,sans-serif;background:#f4f6f8;color:#172033;margin:0}main{max-width:1100px;margin:0 auto;padding:28px}h1{margin:0 0 6px}p{color:#526070}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}.card{background:white;border:1px solid #dce3ea;border-radius:12px;padding:18px;margin:16px 0}label{display:block;font-weight:650;margin:8px 0 4px}input,select,textarea,button{font:inherit;padding:8px;border-radius:7px;border:1px solid #b9c5d1;width:100%;box-sizing:border-box}button{background:#0b6e4f;color:white;border:0;font-weight:700;cursor:pointer;margin-top:12px}.secondary{background:#325c86}.danger{background:#ae3e45}.row{display:flex;gap:10px;align-items:center}.row input[type=checkbox]{width:auto}.status{font-weight:700}.job{border-left:5px solid #0b6e4f;padding:12px;margin:10px 0;background:#f9fbfc}.muted{font-size:13px;color:#68778a}code{background:#edf1f5;padding:2px 4px;border-radius:4px}</style></head>
<body><main><h1>Blender Facade Synthesis Studio</h1><p>Local Web Studio · local serialized BlenderProc Worker · manual publication only.</p>
<div class="card"><div class="row"><button class="secondary" onclick="preflight()">Check BlenderProc runtime</button><span id="preflight" class="muted"></span></div></div>
<section class="card"><h2>Generation Brief</h2><div class="grid">
<div><label>Task dataset</label><select id="task"><option value="window_instance_count">Window instances and count</option><option value="floorline_heatmap">Floorline heatmap</option><option value="visible_floor_count">Visible floor count</option><option value="building_use">Building use</option><option value="facade_component_segmentation">Facade component segmentation</option></select></div>
<div><label>Exact output target</label><input id="count" type="number" min="3" step="3" value="9"></div><div><label>Seed</label><input id="seed" type="number" value="0"></div>
<div><label>Daylight profile</label><select id="daylight" onchange="syncDaylightDistribution()"><option value="daylight_diverse">Daylight diverse</option><option value="controlled_daylight">Controlled clear / overcast</option></select></div>
<div><label>Lighting intensity minimum</label><input id="intensityMin" type="number" min="0.1" max="4" step="0.05" value="0.8"></div><div><label>Lighting intensity maximum</label><input id="intensityMax" type="number" min="0.1" max="4" step="0.05" value="1.2"></div>
<div><label>Task visibility threshold</label><input id="visibility" type="number" min="0" max="1" step="0.05" value="0.50"></div>
<div><label>Resolution width</label><input id="width" type="number" min="32" value="1024"></div><div><label>Resolution height</label><input id="height" type="number" min="32" value="768"></div>
</div><p class="muted">Every Building Recipe emits the full view family (frontal, light/medium-oblique, strong-oblique), so the exact target must be a multiple of 3. Occlusion: clear, 0–15%, 15–30%.</p>
<div class="grid"><div><label>Train split</label><input id="train" type="number" step="0.05" value="0.70"></div><div><label>Validation split</label><input id="validation" type="number" step="0.05" value="0.15"></div><div><label>Test split</label><input id="test" type="number" step="0.05" value="0.15"></div></div>
<div class="grid"><div><label>Residential share</label><input id="residential" type="number" step="0.05" value="0.25"></div><div><label>Office share</label><input id="office" type="number" step="0.05" value="0.25"></div><div><label>Commercial share</label><input id="commercial" type="number" step="0.05" value="0.25"></div><div><label>Mixed-use share</label><input id="mixed" type="number" step="0.05" value="0.25"></div></div>
<div class="grid"><div><label>Clear daylight share</label><input id="clear" type="number" min="0" max="1" step="0.05" value="0.25"></div><div><label>Overcast share</label><input id="overcast" type="number" min="0" max="1" step="0.05" value="0.25"></div><div><label>Warm low-angle share</label><input id="warm" type="number" min="0" max="1" step="0.05" value="0.25"></div><div><label>Backlit share</label><input id="backlit" type="number" min="0" max="1" step="0.05" value="0.25"></div></div>
<div class="grid"><div><label>Clear occlusion share</label><input id="occlusionClear" type="number" min="0" max="1" step="0.05" value="0.3333333333333333"></div><div><label>Light occlusion share</label><input id="occlusionLight" type="number" min="0" max="1" step="0.05" value="0.3333333333333333"></div><div><label>Moderate occlusion share</label><input id="occlusionModerate" type="number" min="0" max="1" step="0.05" value="0.3333333333333333"></div></div>
<label>Optional local PBR/HDRI asset paths (one per line; fingerprinted automatically)</label><textarea id="assets" rows="2"></textarea>
<label class="row"><input id="humanConfirm" type="checkbox"> I have checked output target, angles, lighting, and building-use distribution.</label><button onclick="createAndConfirm()">Confirm Generation Brief and queue job</button><div id="message" class="muted"></div></section>
<section class="card"><h2>Worker queue and manual publication</h2><div id="jobs">Loading…</div></section></main>
<script>
const api=async(url,opts={})=>{const r=await fetch(url,{headers:{'Content-Type':'application/json'},...opts});const v=await r.json();if(!r.ok)throw Error(v.detail||JSON.stringify(v));return v};
const val=id=>document.getElementById(id).value; const number=id=>Number(val(id));
function syncDaylightDistribution(){if(val('daylight')==='controlled_daylight'){document.getElementById('clear').value='0.5';document.getElementById('overcast').value='0.5';document.getElementById('warm').value='0';document.getElementById('backlit').value='0'}}
async function preflight(){try{const v=await api('/api/preflight');document.getElementById('preflight').textContent=`ready: BlenderProc ${v.blenderproc_version}, Blender ${v.blender_version}, NumPy ${v.numpy_version}`}catch(e){document.getElementById('preflight').textContent=`environment not ready: ${e.message}`}}
function payload(){const daylight=val('daylight');const daylightDistribution=daylight==='controlled_daylight'?{clear:number('clear'),overcast:number('overcast')}:{clear:number('clear'),overcast:number('overcast'),warm_low_angle:number('warm'),backlit:number('backlit')};return {task:val('task'),output_target:number('count'),seed:number('seed'),render_width:number('width'),render_height:number('height'),daylight_profile:daylight,daylight_distribution:daylightDistribution,lighting_intensity_range:{min:number('intensityMin'),max:number('intensityMax')},occlusion_distribution:{clear:number('occlusionClear'),light_0_15:number('occlusionLight'),moderate_15_30:number('occlusionModerate')},task_visibility_threshold:number('visibility'),split_ratio:{train:number('train'),validation:number('validation'),test:number('test')},building_use_distribution:{residential:number('residential'),office:number('office'),commercial:number('commercial'),mixed_use:number('mixed')},asset_paths:val('assets').split('\n').map(x=>x.trim()).filter(Boolean)}}
async function createAndConfirm(){try{if(!document.getElementById('humanConfirm').checked)throw Error('Human confirmation is required before queuing.');const j=await api('/api/jobs',{method:'POST',body:JSON.stringify(payload())});await api(`/api/jobs/${j.id}/confirm`,{method:'POST',body:JSON.stringify({confirmed_by:'local-reviewer'})});document.getElementById('message').textContent=`Queued ${j.id}`;loadJobs()}catch(e){document.getElementById('message').textContent=e.message}}
async function act(id,action,body={}){try{await api(`/api/jobs/${id}/${action}`,{method:'POST',body:JSON.stringify(body)});loadJobs()}catch(e){alert(e.message)}}
async function runNext(){try{await api('/api/jobs/run-next',{method:'POST'});loadJobs()}catch(e){alert(e.message)}}
async function loadJobs(){const jobs=await api('/api/jobs');const root=document.getElementById('jobs');root.innerHTML='';if(!jobs.length){root.textContent='No jobs yet.';return}for(const j of jobs){const d=document.createElement('div');d.className='job';let controls='';if(j.state==='queued')controls=`<button onclick="runNext()">Run next serialized job</button>`;if(j.state==='ready_for_review')controls=`<button onclick="act('${j.id}','review',{reviewer:'local-reviewer',approved:true})">Approve review</button>`;if(j.state==='ready_for_review'&&j.review_approved)controls+=`<button onclick="act('${j.id}','publish',{published_by:'local-reviewer'})">Publish package</button>`;if(['queued','draft','running'].includes(j.state))controls+=`<button class="danger" onclick="act('${j.id}','cancel')">Cancel safely</button>`;if(['failed','cancelled'].includes(j.state))controls+=`<button class="secondary" onclick="act('${j.id}','resume')">Resume validated samples</button>`;const pkg=j.package_dir?`<a href="/packages/${j.id}/preview/contact_sheet.png" target="_blank">contact sheet</a> · <a href="/packages/${j.id}/qa_summary.json" target="_blank">QA summary</a>`:'—';d.innerHTML=`<div><code>${j.id}</code> <span class="status">${j.state}</span></div><div class="muted">${j.brief.task} · target ${j.brief.output_target} · valid ${j.validated_sample_count} · ${pkg}</div>${controls}`;root.appendChild(d)}}
loadJobs();setInterval(loadJobs,3000);
</script></body></html>"""
