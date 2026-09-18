import html
import time

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from workflow.graph import MAX_ITERATIONS, build_graph

load_dotenv()

AGENTS = [
    ("planner", "Planner", "Maps the question into evidence tasks"),
    ("researcher", "Researcher", "Collects current web evidence"),
    ("analyst", "Analyst", "Synthesizes findings and tensions"),
    ("critic", "Critic", "Tests coverage and reliability"),
    ("writer", "Writer", "Shapes the validated report"),
]

st.set_page_config(page_title="Research Desk", page_icon="◌", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#f4f7fb; --muted:#8d98a8; --line:rgba(255,255,255,.10); --cyan:#00e5ff; --blue:#3d67ff; --violet:#9a6cff; }
.stApp { background:#050608; color:var(--ink); font-family:'Manrope',sans-serif; }
[data-testid="stHeader"] { background:rgba(5,6,8,.92); }
[data-testid="stSidebar"] { background:#080a0f; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { font-family:'Manrope',sans-serif; }
.block-container { max-width:none; width:100%; padding:0; }
.brand { display:flex; gap:.65rem; align-items:center; font-weight:800; letter-spacing:.02em; }
.brand-mark { width:10px; height:10px; border-radius:50%; background:var(--cyan); box-shadow:0 0 16px var(--cyan); }
.eyebrow { color:var(--cyan); font:500 .7rem 'DM Mono',monospace; letter-spacing:.17em; text-transform:uppercase; }
.hero-title { font-size:clamp(2.5rem,7vw,6.4rem); line-height:.98; letter-spacing:-.055em; margin:.7rem 0 1rem; max-width:850px; }
.hero-copy { color:var(--muted); max-width:630px; font-size:1rem; line-height:1.7; }
.section-rule { border-top:1px solid var(--line); margin:2rem 0 1.2rem; }
.question-label { color:#dbe3ee; font-size:.83rem; font-weight:700; margin-bottom:.4rem; }
textarea { background:#0d1017 !important; color:var(--ink) !important; border:1px solid rgba(0,229,255,.28) !important; border-radius:16px !important; box-shadow:0 0 28px rgba(0,229,255,.08) !important; }
textarea:focus { border-color:var(--cyan) !important; box-shadow:0 0 30px rgba(0,229,255,.17) !important; }
.stButton > button { background:linear-gradient(120deg,#ee489a,#8661ed 52%,#00cde2); color:white; border:0; border-radius:999px; font-weight:800; padding:.72rem 1.35rem; }
.stButton > button:hover { filter:brightness(1.12); transform:translateY(-1px); }
.pipeline {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: .55rem;
    margin: 1rem 0 2rem 32px;
}
.agent { background:linear-gradient(145deg,rgba(255,255,255,.07),rgba(255,255,255,.025)); border:1px solid var(--line); border-radius:14px; padding:1rem; min-height:112px; }
.agent.done { border-color:rgba(44,220,164,.55); }
.agent.active { border-color:rgba(0,229,255,.8); box-shadow:0 0 25px rgba(0,229,255,.12); }
.agent.waiting { opacity:.54; }
.agent-dot { width:9px; height:9px; border-radius:50%; background:#394352; margin-bottom:1rem; }
.done .agent-dot { background:#39dda9; box-shadow:0 0 12px #39dda9; }.active .agent-dot { background:var(--cyan); box-shadow:0 0 14px var(--cyan); }
.agent-name { font-weight:800; font-size:.9rem; }.agent-note { color:var(--muted); font-size:.72rem; line-height:1.4; margin-top:.35rem; }
.report-content {
    max-width:850px;
    margin-left:32px;
}
.report-head { border-bottom:1px solid var(--line); padding:1.2rem 0; margin-bottom:1.5rem; }
.source-list {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
    gap:.65rem;
}
.source-link {
    display:block;
    padding:.8rem 1rem;
    background:#0d1017;
    border:1px solid var(--line);
    border-radius:10px;
    color:#b8f7ff !important;
    font-size:.8rem;
    overflow-wrap:anywhere;
}
.error-box { border:1px solid rgba(255,90,110,.45); background:rgba(130,20,35,.22); border-radius:12px; padding:1rem; color:#ff9da9; }
.prompt-shell, .thinking-shell { width:min(760px,calc(100% - 48px)); max-width:760px; margin:10vh auto 0; }
.prompt-title, .thinking-title { font-size:clamp(2.4rem,6vw,5rem); line-height:1; letter-spacing:-.05em; margin:.8rem 0 1rem; }
[data-testid="stForm"] { width:min(760px,calc(100% - 48px)); max-width:760px; margin:1.5rem auto 0; padding:18px 22px; border:1px solid rgba(168,176,197,.32); border-radius:18px; background:rgba(5,7,12,.72); box-sizing:border-box; }
.thinking-title { color:#eefaff; text-align:center; }
.thinking-shell .hero-copy { margin:0 auto; text-align:center; }
.thinking-dots { color:var(--cyan); letter-spacing:.1em; }
@keyframes revealPrompt { from { opacity:0; transform:translateY(82px); } to { opacity:1; transform:translateY(0); } }
.prompt-shell { animation:revealPrompt .78s cubic-bezier(.2,.8,.2,1) both; }
@media (max-width:760px) { .block-container { padding:1.2rem .9rem 3rem; } .prompt-shell, [data-testid="stForm"] { width:calc(100% - 32px); } [data-testid="stForm"] { padding:14px; } .pipeline { grid-template-columns:1fr; } .agent { min-height:auto; display:flex; align-items:center; gap:.75rem; padding:.75rem; } .agent-dot { margin:0; flex:none; } .hero-title { font-size:clamp(2.55rem,13vw,4rem); } }
</style>
""",
    unsafe_allow_html=True,
)


PARTICLE_INTRO = """
<style>html,body,#root{width:100vw;min-height:100vh;margin:0;background:#050608;overflow:hidden}canvas{display:block;width:100vw;height:100vh;min-height:100px;}.prompt-screen{display:none;position:absolute;inset:0;padding:24px;box-sizing:border-box;align-items:center;justify-content:center;color:#f4f7fb;font-family:Manrope,Arial,sans-serif;background:radial-gradient(ellipse 70% 58% at 50% 58%,rgba(55,61,76,.16),transparent 72%),#050608;}.prompt-screen:after{content:"";position:absolute;inset:22% 8%;background:rgba(115,124,144,.06);filter:blur(58px);pointer-events:none}.prompt-form{position:relative;z-index:1;display:flex;align-items:center;width:min(700px,100%);min-height:150px;box-sizing:border-box;background:rgba(3,5,9,.94);border:1px solid rgba(168,176,197,.62);border-radius:24px;padding:18px 20px 18px 24px;box-shadow:0 0 34px rgba(125,135,158,.14),0 0 0 1px rgba(255,255,255,.05) inset}.prompt-form input{flex:1;min-width:0;background:transparent;border:0;outline:0;color:#fff;font:18px Manrope,Arial,sans-serif;padding:16px 10px}.prompt-form input::placeholder{color:#8b929f}.prompt-form button{width:54px;height:54px;flex:none;border:0;border-radius:50%;background:linear-gradient(135deg,#d84fb9,#7567f7 55%,#12b9dc);color:#fff;font-size:22px;line-height:1;cursor:pointer;box-shadow:0 0 18px rgba(106,113,255,.35)}@media(max-width:560px){.prompt-screen{padding:16px}.prompt-form{min-height:128px;border-radius:20px;padding:14px 14px 14px 18px}.prompt-form input{font-size:16px;padding:12px 6px}.prompt-form button{width:46px;height:46px;font-size:19px}}</style>
<canvas id="particleSphereCanvas"></canvas>
<script>
const canvas=document.getElementById('particleSphereCanvas');
const ctx=canvas.getContext('2d');
let dpr=window.devicePixelRatio||1,viewWidth=390,viewHeight=840;
function resizeCanvas(){const rect=canvas.getBoundingClientRect();viewWidth=rect.width||390;viewHeight=rect.height||840;dpr=window.devicePixelRatio||1;canvas.width=Math.floor(viewWidth*dpr);canvas.height=Math.floor(viewHeight*dpr);ctx.setTransform(dpr,0,0,dpr,0,0)}
resizeCanvas();window.addEventListener('resize',resizeCanvas);
const numParticles=800,particles=[],baseRadius=100,phiGolden=Math.PI*(3-Math.sqrt(5));
for(let i=0;i<numParticles;i++){const yNorm=1-(i/(numParticles-1))*2,radiusAtY=Math.sqrt(Math.max(0,1-yNorm*yNorm)),theta=phiGolden*i,xNorm=Math.cos(theta)*radiusAtY,zNorm=Math.sin(theta)*radiusAtY,rVariation=baseRadius*(.88+Math.random()*.24),rand=Math.random();let color='#00e5ff';if(rand<.35)color='#1541ff';else if(rand<.65)color='#38bdf8';else color='#61dbb4';particles.push({x0:xNorm*rVariation,y0:yNorm*rVariation,z0:zNorm*rVariation,color,size:.8+Math.random()*1.5,baseAlpha:.45+Math.random()*.55,orbitSpeed:.008+Math.random()*.008,phase:Math.random()*Math.PI*2})}
let animId=null,startTime=null,rotY=0,rotX=.2,zoomStartTime=null,isZooming=false,zoomCompleted=false;
const ROTATE_DWELL_MS=1400,ZOOM_DURATION_MS=1600;
function easeInCubic(t){return t*t*t}
function renderFrame(timestamp){if(!startTime)startTime=timestamp;const elapsed=timestamp-startTime;ctx.clearRect(0,0,viewWidth,viewHeight);const centerX=viewWidth/2,centerY=viewHeight/5-25
;if(!isZooming&&elapsed>=ROTATE_DWELL_MS){isZooming=true;zoomStartTime=timestamp}
let zoomProgress=0;if(isZooming){zoomProgress=Math.min(1,(timestamp-zoomStartTime)/ZOOM_DURATION_MS)}const easedZoom=easeInCubic(zoomProgress),cameraZoom=1+easedZoom*8.5,masterFade=Math.max(0,1-Math.pow(zoomProgress,1.35)),currentRotSpeed=.016+easedZoom*.028;rotY+=currentRotSpeed;
if(masterFade>.001){const coreGlowRadius=baseRadius*1.35*(1+easedZoom*2.5),coreGlow=ctx.createRadialGradient(centerX,centerY,0,centerX,centerY,coreGlowRadius),glowAlpha=.32*masterFade*(1-zoomProgress*.7);coreGlow.addColorStop(0,`rgba(0,229,255,${glowAlpha})`);coreGlow.addColorStop(.35,`rgba(21,65,255,${glowAlpha*.75})`);coreGlow.addColorStop(.7,`rgba(139,92,246,${glowAlpha*.3})`);coreGlow.addColorStop(1,'rgba(0,0,0,0)');ctx.fillStyle=coreGlow;ctx.beginPath();ctx.arc(centerX,centerY,coreGlowRadius,0,Math.PI*2);ctx.fill();ctx.save();ctx.translate(centerX,centerY);ctx.rotate(.24);ctx.beginPath();ctx.ellipse(0,0,baseRadius*1.3*cameraZoom,baseRadius*.44*cameraZoom,0,0,Math.PI*2);ctx.strokeStyle=`rgba(0,229,255,${.2*masterFade})`;ctx.lineWidth=1;ctx.stroke();ctx.restore()}
const cosY=Math.cos(rotY),sinY=Math.sin(rotY),cosX=Math.cos(rotX),sinX=Math.sin(rotX),projected=[];
for(let i=0;i<numParticles;i++){const p=particles[i],x1=p.x0*cosY-p.z0*sinY,z1=p.x0*sinY+p.z0*cosY,y2=p.y0*cosX-z1*sinX,z2=p.y0*sinX+z1*cosX,dist=340/(340+z2*cameraZoom),projX=centerX+x1*cameraZoom*dist,projY=centerY+y2*cameraZoom*dist,depthScale=Math.max(.2,(z2+baseRadius*1.4)/(baseRadius*2.8)),particleAlpha=Math.max(0,Math.min(1,p.baseAlpha*depthScale*masterFade));if(particleAlpha>.005)projected.push({x:projX,y:projY,z:z2,size:Math.max(.5,p.size*cameraZoom*.55*(.8+depthScale*.5)),alpha:particleAlpha,color:p.color})}
projected.sort((a,b)=>a.z-b.z);for(const pt of projected){ctx.beginPath();ctx.arc(pt.x,pt.y,pt.size,0,Math.PI*2);ctx.fillStyle=pt.color;ctx.globalAlpha=pt.alpha;ctx.shadowColor=pt.color;ctx.shadowBlur=pt.z>0&&masterFade>.5?4:0;ctx.fill()}ctx.shadowBlur=0;ctx.globalAlpha=1;
if(isZooming&&zoomProgress>=1&&!zoomCompleted){zoomCompleted=true;cancelAnimationFrame(animId);return}animId=requestAnimationFrame(renderFrame)}
animId=requestAnimationFrame(renderFrame);
</script>
"""


def render_pipeline(completed: set[str], active: str | None = None) -> None:
    cards = []
    for key, name, note in AGENTS:
        state = "done" if key in completed else "active" if key == active else "waiting"
        label = "Complete" if state == "done" else "Running" if state == "active" else "Waiting"
        cards.append(
            f'<div class="agent {state}"><div class="agent-dot"></div><div><div class="agent-name">{name}</div><div class="agent-note">{label} · {note}</div></div></div>'
        )
    st.markdown(f'<div class="pipeline">{"".join(cards)}</div>', unsafe_allow_html=True)


if st.query_params.get("phase") == "thinking":
    question_from_url = st.query_params.get("q", "").strip()
    if question_from_url:
        st.session_state.question = question_from_url
        st.session_state.screen = "thinking"
        st.query_params.clear()
elif st.query_params.get("phase") == "prompt":
    st.session_state.screen = "prompt"
    st.query_params.clear()
if "screen" not in st.session_state:
    st.session_state.screen = "intro"


@st.fragment(run_every="3s")
def render_intro() -> None:
    if "intro_started_at" not in st.session_state:
        st.session_state.intro_started_at = time.monotonic()
    if time.monotonic() - st.session_state.intro_started_at >= 3.2:
        render_prompt()
        return
    components.html(PARTICLE_INTRO, height=1200, scrolling=False)


def render_prompt() -> None:
    st.markdown('<div class="prompt-shell"><h1 class="prompt-title">RESEARCH DESK</h1><p class="hero-copy">Ask a question that benefits from current evidence, comparison, and critical review.</p>', unsafe_allow_html=True)
    with st.form("research_form", clear_on_submit=False):
        question = st.text_area("Question", value=st.session_state.get("question", ""), placeholder="Ask a question", height=150, label_visibility="collapsed")
        submitted = st.form_submit_button("Send question  ↑")
    st.markdown('</div>', unsafe_allow_html=True)
    if submitted:
        if not question.strip():
            st.markdown('<div class="error-box">Enter a research question before starting.</div>', unsafe_allow_html=True)
        else:
            st.session_state.question = question.strip()
            st.session_state.screen = "thinking"
            st.rerun()


def render_thinking() -> None:
    st.markdown('<div class="thinking-shell"><div class="eyebrow">Live execution</div><h1 class="thinking-title">Thinking<span class="thinking-dots">...</span></h1><p class="hero-copy">The pipeline is running against the question you submitted. Status changes come from the real LangGraph nodes.</p></div>', unsafe_allow_html=True)
    pipeline = st.empty()
    status = st.empty()
    completed: set[str] = set()
    final_state: dict = {}
    node_names = {key for key, _, _ in AGENTS}
    try:
        for update in build_graph().stream(
            {"user_question": st.session_state.question, "max_iterations": MAX_ITERATIONS},
            stream_mode="updates",
        ):
            for node_name, node_update in update.items():
                final_state.update(node_update)
                if node_name in node_names:
                    completed.add(node_name)
                    cards = []
                    for key, name, note in AGENTS:
                        state = "done" if key in completed else "waiting"
                        cards.append(f'<div class="agent {state}"><div class="agent-dot"></div><div><div class="agent-name">{name}</div><div class="agent-note">{"Complete" if state == "done" else "Waiting"} · {note}</div></div></div>')
                    pipeline.markdown(f'<div class="pipeline">{"".join(cards)}</div>', unsafe_allow_html=True)
                    status.info(final_state.get("progress_log", [f"{node_name} complete"])[-1])
        report = final_state.get("final_report")
        if not report:
            raise RuntimeError("The Writer completed without producing a final report.")
        st.session_state.result = final_state
        st.session_state.screen = "report"
        st.rerun()
    except Exception as exc:
        st.session_state.screen = "error"
        st.session_state.error = str(exc)
        st.rerun()


def render_report() -> None:
    result = st.session_state.result
    st.markdown('<div class="report-head" style="margin-left:32px;"><div class="eyebrow">Validated research report</div><h2>Executive synthesis</h2></div>', unsafe_allow_html=True)
    st.markdown(
    f'<div class="report-content">{result["final_report"]}</div>',
    unsafe_allow_html=True,
)
    sources, seen_urls = [], set()
    for source in result.get("sources", []):
        if source.url not in seen_urls:
            seen_urls.add(source.url)
            sources.append(source)
  
    if sources:
      st.markdown(
        '<div style="margin-left:32px;"><div class="section-rule"></div><div class="eyebrow">Evidence ledger</div><h3>Sources used</h3></div>',
        unsafe_allow_html=True
    )

      st.markdown(
        '<div class="source-list" style="margin-left:32px;">' +
        "".join(
            f'<a class="source-link" href="{html.escape(source.url)}" target="_blank">{html.escape(source.title)}</a>'
            for source in sources
        ) +
        '</div>',
        unsafe_allow_html=True
    )
st.markdown("<div style='margin-left:32px;height:50px;'>", unsafe_allow_html=True)
if st.button("Research another question  ↗"):
    st.session_state.screen = "prompt"
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


def render_error() -> None:
    st.markdown(f'<div class="error-box"><strong>Research could not be completed.</strong><br>{html.escape(st.session_state.get("error", "Unknown workflow error"))}</div>', unsafe_allow_html=True)
    if st.button("Return to prompt"):
        st.session_state.screen = "prompt"
        st.rerun()


if st.session_state.screen == "intro":
    render_intro()
elif st.session_state.screen == "prompt":
    render_prompt()
elif st.session_state.screen == "thinking":
    render_thinking()
elif st.session_state.screen == "report":
    render_report()
else:
    render_error()
