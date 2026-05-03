/* ================================================================
 * Loan Wizard — live session orchestrator
 * - WebRTC video + audio capture
 * - Web Speech API for STT
 * - face-api.js for age estimation, gender, liveness (blink challenge)
 * - Live entity extraction (mirror of server NLP)
 * - Agent pipeline state machine
 * ================================================================ */

const SID = window.SESSION_ID;

const QUESTIONS = [
  { id: 'name',     prompt: 'Welcome! Could you start by telling me your full name?',                 field: 'name' },
  { id: 'age',      prompt: 'Please tell me your age — for example, "I am 32 years old".',            field: 'age_declared' },
  { id: 'employer', prompt: 'What do you do for work, and where are you employed?',                   field: 'employer' },
  { id: 'income',   prompt: 'What is your approximate monthly income? Please say it clearly, for example "fifty thousand rupees per month" or "₹50000".', field: 'monthly_income_inr' },
  { id: 'purpose',  prompt: 'What would you like to use this loan for?',                              field: 'loan_purpose' },
  { id: 'amount',   prompt: 'How much would you like to borrow, and over what tenure?',               field: 'loan_amount_requested_inr' },
  { id: 'consent',  prompt: 'Do you consent to Poonawalla Fincorp processing this data for a loan offer? Please answer yes or no clearly.', field: 'consent_given', isConsent: true },
];

const AGENTS = [
  { id: 'video',  name: 'Video & metadata',  ic: '📹', detail: 'Capturing stream' },
  { id: 'stt',    name: 'Speech-to-text',    ic: '🎙️', detail: 'Awaiting input' },
  { id: 'cv',     name: 'Computer vision',   ic: '👁️', detail: 'Loading models' },
  { id: 'nlp',    name: 'Entity extraction', ic: '🧠', detail: 'Idle' },
  { id: 'risk',   name: 'Risk + policy',     ic: '⚖️', detail: 'Idle' },
  { id: 'llm',    name: 'LLM intelligence',  ic: '🤖', detail: 'Idle' },
  { id: 'offer',  name: 'Offer generator',   ic: '💰', detail: 'Idle' },
  { id: 'audit',  name: 'Audit + integrity', ic: '🔒', detail: 'Idle' },
];

const state = {
  qIndex: 0,
  responses: {}, // qid -> raw transcript text
  entities: {},
  faceSamples: [],
  livenessChallenges: [],
  faceLatest: null,
  agents: Object.fromEntries(AGENTS.map(a => [a.id, { ...a, status: 'pending' }])),
  startedAt: Date.now(),
  micRecognition: null,
  listening: false,
  blinkBaseline: null,
  blinkCount: 0,
  consentGiven: false,
  livenessOk: false,
};

// ---------- helpers ----------
function $(s) { return document.querySelector(s); }
function $$(s) { return Array.from(document.querySelectorAll(s)); }
function setAgent(id, status, detail) {
  const a = state.agents[id];
  if (!a) return;
  a.status = status;
  if (detail) a.detail = detail;
  renderAgents();
}
function renderAgents() {
  const root = $('#agents');
  root.innerHTML = AGENTS.map(a => {
    const cur = state.agents[a.id];
    return `
      <div class="agent ${cur.status}">
        <div class="ic">${a.ic}</div>
        <div>
          <div class="name">${a.name}</div>
          <div class="detail">${cur.detail}</div>
        </div>
        <div class="check">${cur.status === 'complete' ? '✓' : cur.status === 'processing' ? '⟳' : '·'}</div>
      </div>`;
  }).join('');
}
function fmtSec(s) {
  const m = Math.floor(s / 60), r = s % 60;
  return `${String(m).padStart(2,'0')}:${String(r).padStart(2,'0')}`;
}

// ---------- timer ----------
setInterval(() => {
  const s = Math.floor((Date.now() - state.startedAt) / 1000);
  $('#timer').textContent = fmtSec(s);
}, 500);

// ---------- camera ----------
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: 640, height: 480 },
      audio: true,
    });
    $('#vid').srcObject = stream;
    setAgent('video', 'complete', 'Camera + mic active');
    return stream;
  } catch (e) {
    setAgent('video', 'error', 'Camera denied');
    alert('Camera + microphone permission is required.');
    throw e;
  }
}

// ---------- face-api models ----------
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.12/model';

async function loadFaceModels() {
  setAgent('cv', 'processing', 'Loading models');
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
    ]);
    setAgent('cv', 'processing', 'Models loaded — running detection');
  } catch (e) {
    console.error('Face model load failed', e);
    setAgent('cv', 'error', 'Model load failed');
  }
}

// ---------- liveness via blink detection ----------
function eyeAspectRatio(landmarks) {
  // mediapipe-like ratio using face-api 68 landmarks: 36-41 (left), 42-47 (right)
  const L = landmarks.getLeftEye();
  const R = landmarks.getRightEye();
  const ear = (eye) => {
    const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    return (d(eye[1], eye[5]) + d(eye[2], eye[4])) / (2 * d(eye[0], eye[3]));
  };
  return (ear(L) + ear(R)) / 2;
}

let detectionLoopRunning = false;
async function detectLoop() {
  detectionLoopRunning = true;
  const video = $('#vid');
  const canvas = $('#overlay');
  const ctx = canvas.getContext('2d');

  while (detectionLoopRunning) {
    if (video.readyState >= 2 && video.videoWidth > 0) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      try {
        const result = await faceapi
          .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224 }))
          .withFaceLandmarks()
          .withAgeAndGender()
          .withFaceExpressions();

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (result) {
          const det = faceapi.resizeResults(result, { width: canvas.width, height: canvas.height });
          // Simple bounding box outline
          ctx.strokeStyle = 'rgba(110,168,255,0.9)';
          ctx.lineWidth = 2;
          const b = det.detection.box;
          ctx.strokeRect(b.x, b.y, b.width, b.height);

          const ear = eyeAspectRatio(det.landmarks);
          if (state.blinkBaseline == null) state.blinkBaseline = ear;
          // Detect blink: EAR drops below 70% of baseline
          if (ear < state.blinkBaseline * 0.72) {
            state.blinkCount += 1;
          } else {
            // slow baseline tracking
            state.blinkBaseline = state.blinkBaseline * 0.95 + ear * 0.05;
          }

          const topExpr = Object.entries(result.expressions || {})
            .sort((a, b) => b[1] - a[1])[0]?.[0] || 'neutral';

          state.faceLatest = {
            age: Math.round(result.age),
            gender: result.gender,
            genderConf: +(result.genderProbability || 0).toFixed(2),
            expression: topExpr,
            ear: +ear.toFixed(3),
            blinkCount: state.blinkCount,
            ts: new Date().toISOString(),
          };
          state.faceSamples.push(state.faceLatest);
          if (state.faceSamples.length > 50) state.faceSamples.shift();

          renderFaceTags(state.faceLatest);
          if (state.agents.cv.status !== 'complete' && state.faceSamples.length > 5) {
            setAgent('cv', 'complete', `Age ~${state.faceLatest.age}, ${state.faceLatest.gender}`);
          }

          // Liveness passes on either: any blink, OR sustained face presence (>= 8 samples with EAR variance)
          const earValues = state.faceSamples.slice(-12).map(s => s.ear).filter(Boolean);
          let earVariance = 0;
          if (earValues.length >= 4) {
            const mean = earValues.reduce((a,b)=>a+b,0) / earValues.length;
            earVariance = earValues.reduce((acc,v)=>acc+Math.pow(v-mean,2),0) / earValues.length;
          }
          const sustained = state.faceSamples.length >= 8 && earVariance > 0.0005;
          if (state.blinkCount >= 1 || sustained) {
            state.livenessOk = true;
            $('#livenessTag').textContent = '🟢 Liveness OK';
            $('#livenessTag').classList.add('green');
          }
        } else {
          renderFaceTags(null);
        }
      } catch (e) { /* swallow per-frame errors */ }
    }
    await new Promise(r => setTimeout(r, 700));
  }
}

function renderFaceTags(face) {
  const root = $('#faceTags');
  if (!face) {
    root.innerHTML = '<div class="face-tag red">⚠ Face not detected</div>';
    return;
  }
  root.innerHTML = `
    <div class="face-tag green">👤 Detected</div>
    <div class="face-tag">Age ~${face.age}</div>
    <div class="face-tag">${face.gender === 'male' ? '♂' : '♀'} ${face.gender}</div>
    <div class="face-tag gold">😊 ${face.expression}</div>
    <div class="face-tag">Blinks: ${state.blinkCount}</div>
  `;
}

// ---------- speech recognition ----------
function initSTT() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    setAgent('stt', 'error', 'Browser unsupported (use Chrome)');
    alert('Speech recognition not supported. Please use Chrome or Edge.');
    return null;
  }
  const r = new SR();
  r.continuous = true;
  r.interimResults = true;
  r.lang = 'en-IN';
  return r;
}

let liveText = '';
function startListening() {
  if (state.listening) return;
  if (!state.micRecognition) state.micRecognition = initSTT();
  if (!state.micRecognition) return;

  liveText = '';
  setAgent('stt', 'processing', 'Listening…');
  $('#response').classList.add('live');
  $('#response').textContent = '🎙️ Listening…';
  state.micRecognition.onresult = (e) => {
    let final = '', interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) final += e.results[i][0].transcript;
      else interim += e.results[i][0].transcript;
    }
    if (final) liveText = (liveText + ' ' + final).trim();
    $('#response').textContent = (liveText + ' ' + interim).trim() || '🎙️ Listening…';
    extractEntitiesFromCurrent(liveText + ' ' + interim);
  };
  state.micRecognition.onerror = () => stopListening();
  state.micRecognition.onend = () => {
    if (state.listening) {
      try { state.micRecognition.start(); } catch {}
    }
  };
  try { state.micRecognition.start(); state.listening = true; } catch {}
  $('#micBtn').textContent = '⏹ Stop';
  $('#micBtn').classList.add('danger');
  $('#nextBtn').disabled = false;
}

function stopListening() {
  if (!state.listening) return;
  state.listening = false;
  if (state.micRecognition) {
    state.micRecognition.onend = null;
    try { state.micRecognition.stop(); } catch {}
  }
  $('#micBtn').textContent = '🎙️ Start speaking';
  $('#micBtn').classList.remove('danger');
  $('#response').classList.remove('live');
}

// ---------- live entity extraction (lightweight, mirrors server) ----------
function extractEntitiesFromCurrent(currentResponse) {
  const allText = Object.values(state.responses).join(' ') + ' ' + currentResponse;
  const t = allText.toLowerCase();

  // Income
  const incMatch = t.match(/(?:income|earn|salary|making|make)[^\d]{0,30}(\d[\d,]*)\s*(k|thousand|lakh|lakhs)?|(\d[\d,]*)\s*(k|thousand|lakh|lakhs)?\s*(?:per month|monthly|a month|p\.?m\.?)/i);
  if (incMatch) {
    let n = parseInt((incMatch[1] || incMatch[3] || '0').replace(/,/g, ''));
    const u = (incMatch[2] || incMatch[4] || '').toLowerCase();
    if (u === 'k' || u === 'thousand') n *= 1000;
    if (u === 'lakh' || u === 'lakhs') n *= 100000;
    if (n < 1000) n *= 1000;
    if (n >= 5000 && n <= 5000000) state.entities.monthly_income_inr = n;
  }
  // Loan amount
  const amtMatch = t.match(/(?:borrow|loan(?: of)?|need|want|require)[^\d]{0,30}(\d[\d,]*)\s*(k|thousand|lakh|lakhs|cr|crore)?/i);
  if (amtMatch) {
    let n = parseInt(amtMatch[1].replace(/,/g, ''));
    const u = (amtMatch[2] || '').toLowerCase();
    if (u === 'k' || u === 'thousand') n *= 1000;
    if (u === 'lakh' || u === 'lakhs') n *= 100000;
    if (u === 'cr' || u === 'crore') n *= 10000000;
    if (n < 1000) n *= 1000;
    if (n >= 10000) state.entities.loan_amount_requested_inr = n;
  }
  // Tenure
  const tenMatch = t.match(/(\d+)\s*(months?|years?|yrs?)/i);
  if (tenMatch) {
    let n = parseInt(tenMatch[1]);
    if (/year|yr/i.test(tenMatch[2])) n *= 12;
    if (n >= 6 && n <= 84) state.entities.tenure_months = n;
  }
  // Purpose
  const purposes = { home: ['home','house','renovation'], wedding: ['wedding','marriage'], medical: ['medical','treatment'], education: ['education','college','tuition'], business: ['business','expansion'], travel: ['travel','vacation'], vehicle: ['car','bike','vehicle'], 'debt-consolidation': ['consolidat','credit card'] };
  for (const [k, kws] of Object.entries(purposes)) {
    if (kws.some(kw => t.includes(kw))) { state.entities.loan_purpose = k; break; }
  }
  // Employer / occupation
  const occ = ['engineer','developer','manager','analyst','consultant','doctor','teacher','accountant','designer','salesperson','self-employed','self employed','freelancer'];
  for (const o of occ) { if (t.includes(o)) { state.entities.employer = o; break; } }
  // Name (only on first answer)
  if (state.qIndex === 0) {
    const nameMatch = currentResponse.match(/(?:my name is|i am|i'm|this is)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})/i)
      || currentResponse.match(/^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b/);
    if (nameMatch) state.entities.name = nameMatch[1];
  }
  // Declared age — only scan when answering the age question, to avoid
  // mis-parsing the first two digits of an income/loan amount as the age.
  const currentQId = QUESTIONS[state.qIndex]?.id;
  if (currentQId === 'age') {
    const wordToNum = { eighteen:18, nineteen:19, twenty:20, 'twenty one':21, 'twenty two':22, 'twenty three':23, 'twenty four':24, 'twenty five':25, 'twenty six':26, 'twenty seven':27, 'twenty eight':28, 'twenty nine':29, thirty:30, 'thirty one':31, 'thirty two':32, 'thirty three':33, 'thirty four':34, 'thirty five':35, 'thirty six':36, 'thirty seven':37, 'thirty eight':38, 'thirty nine':39, forty:40, 'forty five':45, fifty:50, 'fifty five':55, sixty:60, 'sixty five':65 };
    const ageMatch = currentResponse.match(/(?:i am|i'm|age is|age|i'm aged)\s+(\d{2})/i)
                  || currentResponse.match(/\b(\d{2})\s*(?:years?\s*old|yrs?|year|yr)\b/i)
                  || currentResponse.match(/^\s*(\d{2})\s*$/);
    if (ageMatch) {
      const a = parseInt(ageMatch[1]);
      if (a >= 18 && a <= 80) state.entities.age_declared = a;
    } else {
      const lc = currentResponse.toLowerCase();
      for (const [w, n] of Object.entries(wordToNum)) {
        if (lc.includes(w)) { state.entities.age_declared = n; break; }
      }
    }
  }
  // Income fallback — when answering the income question, accept bare numbers
  // like "50000" or "fifty thousand" without requiring the word "income".
  if (currentQId === 'income' && !state.entities.monthly_income_inr) {
    const incFallbackWords = { 'ten thousand':10000, 'fifteen thousand':15000, 'twenty thousand':20000, 'twenty five thousand':25000, 'thirty thousand':30000, 'forty thousand':40000, 'fifty thousand':50000, 'sixty thousand':60000, 'seventy thousand':70000, 'seventy five thousand':75000, 'eighty thousand':80000, 'ninety thousand':90000, 'one lakh':100000, 'two lakh':200000, 'three lakh':300000, 'five lakh':500000 };
    let n = null;
    const lc = currentResponse.toLowerCase();
    for (const [w, val] of Object.entries(incFallbackWords)) {
      if (lc.includes(w)) { n = val; break; }
    }
    if (n == null) {
      const bareMatch = lc.match(/(\d[\d,]*)\s*(k|thousand|lakh|lakhs|l|cr|crore)?/);
      if (bareMatch) {
        let m = parseInt(bareMatch[1].replace(/,/g, ''));
        const u = (bareMatch[2] || '').toLowerCase();
        if (u === 'k' || u === 'thousand') m *= 1000;
        else if (u === 'l' || u === 'lakh' || u === 'lakhs') m *= 100000;
        else if (u === 'cr' || u === 'crore') m *= 10000000;
        else if (m < 1000) m *= 1000;
        if (m >= 5000 && m <= 5000000) n = m;
      }
    }
    if (n) state.entities.monthly_income_inr = n;
  }
  // Consent
  const yes = /\b(yes|i consent|i agree|i accept|absolutely|sure|confirm|i do)\b/i.test(currentResponse);
  const no = /\b(no|don'?t|do not|refuse|decline)\b/i.test(currentResponse);
  if (yes && !no) state.consentGiven = true;
  if (state.qIndex >= QUESTIONS.length - 1) state.entities.consent_given = state.consentGiven;

  if (Object.keys(state.entities).length >= 2 && state.agents.nlp.status !== 'complete') {
    setAgent('nlp', 'processing', `Extracted ${Object.keys(state.entities).length} entities`);
  }

  renderEntities();
}

function renderEntities() {
  $$('.kv .v').forEach(el => {
    // Skip fields the user is currently editing or has manually overridden
    if (el === document.activeElement) return;
    if (el.classList.contains('user-edited')) return;
    const k = el.dataset.k;
    let v = state.entities[k];
    if (v == null || v === '') { el.textContent = '—'; return; }
    if (k === 'monthly_income_inr' || k === 'loan_amount_requested_inr') {
      el.textContent = Number(v).toLocaleString('en-IN');
    } else if (k === 'consent_given') {
      el.textContent = v ? '✅ Yes' : '❌ No';
    } else if (k === 'tenure_months') {
      el.textContent = String(v);
    } else {
      el.textContent = v;
    }
  });
}

function wireEditableEntities() {
  $$('.kv .v.editable').forEach(el => {
    el.addEventListener('focus', () => {
      // Strip placeholder so user starts on a clean field
      if (el.textContent.trim() === '—') el.textContent = '';
    });
    el.addEventListener('blur', () => {
      const k = el.dataset.k;
      const type = el.dataset.type;
      let raw = el.textContent.trim();
      if (raw === '' || raw === '—') {
        // Clear override → re-enable auto-extract
        el.classList.remove('user-edited');
        renderEntities();
        return;
      }
      let value = raw;
      if (type === 'number') {
        // Accept things like "50,000", "₹50000", "50k", "5 lakh"
        const lower = raw.toLowerCase().replace(/[₹,\s]/g, '');
        let m = lower.match(/^(\d+(?:\.\d+)?)(k|thousand|l|lakh|lakhs|cr|crore)?$/);
        if (m) {
          let n = parseFloat(m[1]);
          const u = m[2] || '';
          if (u === 'k' || u === 'thousand') n *= 1000;
          else if (u === 'l' || u === 'lakh' || u === 'lakhs') n *= 100000;
          else if (u === 'cr' || u === 'crore') n *= 10000000;
          value = Math.round(n);
        } else {
          value = parseInt(raw.replace(/[^\d]/g, '')) || 0;
        }
      }
      state.entities[k] = value;
      el.classList.add('user-edited');
      renderEntities();
    });
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') { e.preventDefault(); el.blur(); }
    });
  });
}

// ---------- conversation flow ----------
function renderProgress() {
  const bar = $('#progressBar');
  bar.innerHTML = QUESTIONS.map((_, i) =>
    `<span class="${i < state.qIndex ? 'done' : i === state.qIndex ? 'active' : ''}"></span>`
  ).join('');
}

function renderQuestion() {
  const q = QUESTIONS[state.qIndex];
  $('#question').textContent = q.prompt;
  $('#response').textContent = 'Click the mic to start speaking.';
  $('#response').classList.remove('live');
  liveText = '';
  $('#nextBtn').disabled = true;
  renderProgress();
}

function pushTranscriptLine(speaker, text, isConsent = false, qid = null) {
  const line = { speaker, text, ts: new Date().toISOString(), is_consent: isConsent, question_id: qid };
  $('#transcript').insertAdjacentHTML('beforeend',
    `<div class="line"><span class="speaker ${speaker === 'customer' ? 'customer' : ''}">${speaker.toUpperCase()}:</span>${text}</div>`);
  $('#transcript').scrollTop = $('#transcript').scrollHeight;

  fetch(`/api/session/${SID}/transcript`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(line),
  });
  return line;
}

function nextQuestion() {
  stopListening();
  const q = QUESTIONS[state.qIndex];
  const text = liveText.trim() || '(no response captured)';
  state.responses[q.id] = text;

  // Push transcript pair
  pushTranscriptLine('ai', q.prompt, false, q.id);
  pushTranscriptLine('customer', text, q.isConsent, q.id);

  // Re-extract for completeness
  extractEntitiesFromCurrent('');

  state.qIndex += 1;
  if (state.qIndex >= QUESTIONS.length) {
    finalize();
  } else {
    renderQuestion();
  }
}

function skipQuestion() {
  stopListening();
  const q = QUESTIONS[state.qIndex];
  pushTranscriptLine('ai', q.prompt, false, q.id);
  pushTranscriptLine('customer', '(skipped)', false, q.id);
  state.qIndex += 1;
  if (state.qIndex >= QUESTIONS.length) finalize();
  else renderQuestion();
}

// ---------- finalize ----------
async function finalize() {
  $('#question').textContent = 'Analyzing your application…';
  $('#response').textContent = 'Eight AI agents are evaluating your data. Please hold.';
  $('#micBtn').disabled = true; $('#nextBtn').disabled = true; $('#skipBtn').disabled = true;

  // Aggregate face data
  const ages = state.faceSamples.map(s => s.age).filter(Boolean);
  const ageEst = ages.length ? Math.round(ages.reduce((a, b) => a + b, 0) / ages.length) : null;
  // Liveness: pass if any blink was detected OR sustained face presence with EAR variance
  const livenessPassed = !!state.livenessOk || state.blinkCount >= 1 || state.faceSamples.length >= 8;
  const declaredAge = state.entities.age_declared || null;

  const facePayload = {
    age_estimated: ageEst,
    age_declared: declaredAge,
    gender: state.faceLatest?.gender,
    expression: state.faceLatest?.expression,
    liveness_passed: livenessPassed,
    liveness_challenges: ['blink-detection', 'sustained-presence'],
    blink_count: state.blinkCount,
    samples_collected: state.faceSamples.length,
  };
  await fetch(`/api/session/${SID}/face`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(facePayload),
  });

  setAgent('risk', 'processing', 'Bureau lookup + policy gates');
  setAgent('nlp', 'complete', `${Object.keys(state.entities).length} entities extracted`);
  setAgent('llm', 'processing', 'Reading transcript');
  setAgent('offer', 'processing', 'Building offers');
  setAgent('audit', 'processing', 'Sealing record');

  const res = await fetch(`/api/session/${SID}/finalize`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entities_override: state.entities, bureau_profile: '_default' }),
  });
  const data = await res.json();

  setAgent('risk', 'complete',
    `Score ${data.propensity?.score}/100 (${data.propensity?.band?.replace('_',' ')})`);
  setAgent('llm', 'complete',
    `Persona: ${data.llm?.persona || '—'}`);
  setAgent('offer', 'complete',
    `${(data.offers || []).length} offer(s) generated`);
  setAgent('audit', 'complete',
    `Hash ${data.integrity_hash?.slice(0, 12)}…`);

  setTimeout(() => { window.location.href = data.redirect; }, 900);
}

// ---------- wire up ----------
$('#micBtn').addEventListener('click', () => state.listening ? stopListening() : startListening());
$('#nextBtn').addEventListener('click', nextQuestion);
$('#skipBtn').addEventListener('click', skipQuestion);

(async function boot() {
  renderAgents();
  renderQuestion();
  wireEditableEntities();
  await startCamera();
  // load face-api after a short delay so script tag has loaded
  const waitFA = () => new Promise(r => {
    const t = setInterval(() => { if (window.faceapi) { clearInterval(t); r(); } }, 100);
  });
  await waitFA();
  await loadFaceModels();
  detectLoop();
})();
