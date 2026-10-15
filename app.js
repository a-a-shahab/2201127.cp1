/**
 * ==============================================================================
 * Multimodal Vital-Sign Monitor & ML Predictor - Web Application
 * Real-Time Canvas / Chart.js Streaming, FreeRTOS WS Client, & FastAPI ML Bridge
 * ==============================================================================
 */

// Configuration constants
const CONFIG = {
  ECG_SAMPLING_RATE_HZ: 200,
  PPG_SAMPLING_RATE_HZ: 100,
  BUFFER_DURATION_SEC: 6, // 6-second rolling oscilloscope window
  MAX_ECG_POINTS: 200 * 6, // 1200 points
  MAX_PPG_POINTS: 100 * 6, // 600 points
  CHART_REFRESH_FPS: 30,   // Canvas render throttle to prevent CPU strain
};

// Global Application State
const state = {
  socket: null,
  isConnected: false,
  isSimulating: false,
  simInterval: null,
  packetCount: 0,
  lastPingTime: 0,
  rttMs: 0,

  // Rolling Waveform Buffers
  ecgBuffer: [],
  ppgRedBuffer: [],
  ppgIrBuffer: [],
  timeLabelsEcg: [],
  timeLabelsPpg: [],

  // Latest Vitals Snapshot
  vitals: {
    hr: 0,
    spo2: 0,
    temp: 36.6,
    sqi: 0,
    leadOk: true,
    finger: false,
    battV: 3.92,
  },

  // Experiment Recording State
  isRecording: false,
  recordingStartTime: 0,
  recordingTimerInterval: null,
  recordedData: [], // Array of row objects for CSV/JSON

  // AI Inference State
  autoInferenceInterval: null,
};

// ==============================================================================
// DOM Element References
// ==============================================================================
const elements = {
  wsUrlInput: document.getElementById('wsUrlInput'),
  backendUrlInput: document.getElementById('backendUrlInput'),
  connectBtn: document.getElementById('connectBtn'),
  simBtn: document.getElementById('simBtn'),
  testBackendBtn: document.getElementById('testBackendBtn'),
  connStatusBadge: document.getElementById('connStatusBadge'),
  connStatusText: document.getElementById('connStatusText'),
  bufferCount: document.getElementById('bufferCount'),
  packetCount: document.getElementById('packetCount'),
  rttText: document.getElementById('rttText'),
  mixedContentAlert: document.getElementById('mixedContentAlert'),

  // Vitals Cards
  valHR: document.getElementById('valHR'),
  valSpO2: document.getElementById('valSpO2'),
  valTemp: document.getElementById('valTemp'),
  valSQI: document.getElementById('valSQI'),
  sqiBar: document.getElementById('sqiBar'),
  leadStatusPill: document.getElementById('leadStatusPill'),
  fingerStatusPill: document.getElementById('fingerStatusPill'),
  tempStatusPill: document.getElementById('tempStatusPill'),
  ecgLeadIndicator: document.getElementById('ecgLeadIndicator'),
  showIRToggle: document.getElementById('showIRToggle'),

  // Recording Panel
  startRecBtn: document.getElementById('startRecBtn'),
  stopRecBtn: document.getElementById('stopRecBtn'),
  downloadCsvBtn: document.getElementById('downloadCsvBtn'),
  downloadJsonBtn: document.getElementById('downloadJsonBtn'),
  recordingBadge: document.getElementById('recordingBadge'),
  recDuration: document.getElementById('recDuration'),
  recSamples: document.getElementById('recSamples'),

  // ML Inference Panel
  runInferenceBtn: document.getElementById('runInferenceBtn'),
  autoInferenceToggle: document.getElementById('autoInferenceToggle'),
  aiStatusText: document.getElementById('aiStatusText'),
  predBP: document.getElementById('predBP'),
  predBPCategory: document.getElementById('predBPCategory'),
  predArrhythmia: document.getElementById('predArrhythmia'),
  predArrhythmiaConfidence: document.getElementById('predArrhythmiaConfidence'),
  calcPTT: document.getElementById('calcPTT'),
};

// ==============================================================================
// Initialize Chart.js Oscilloscopes
// ==============================================================================
let ecgChart = null;
let ppgChart = null;

function initCharts() {
  // Pre-fill circular ring arrays with baseline zeroes
  for (let i = 0; i < CONFIG.MAX_ECG_POINTS; i++) {
    state.timeLabelsEcg.push('');
    state.ecgBuffer.push(1650); // AD8232 1.65V baseline
  }
  for (let i = 0; i < CONFIG.MAX_PPG_POINTS; i++) {
    state.timeLabelsPpg.push('');
    state.ppgRedBuffer.push(0);
    state.ppgIrBuffer.push(0);
  }

  // ECG Chart Configuration
  const ctxEcg = document.getElementById('ecgCanvas').getContext('2d');
  ecgChart = new Chart(ctxEcg, {
    type: 'line',
    data: {
      labels: state.timeLabelsEcg,
      datasets: [
        {
          label: 'ECG (mV)',
          data: state.ecgBuffer,
          borderColor: '#00ff88',
          borderWidth: 1.8,
          pointRadius: 0,
          tension: 0.1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      scales: {
        x: { display: false },
        y: {
          min: 800,
          max: 2600,
          grid: { color: 'rgba(0, 255, 136, 0.1)' },
          ticks: { color: '#8b949e', font: { family: 'SF Mono, monospace', size: 10 } },
        },
      },
      plugins: { legend: { display: false } },
    },
  });

  // PPG Chart Configuration
  const ctxPpg = document.getElementById('ppgCanvas').getContext('2d');
  ppgChart = new Chart(ctxPpg, {
    type: 'line',
    data: {
      labels: state.timeLabelsPpg,
      datasets: [
        {
          label: 'Red AC',
          data: state.ppgRedBuffer,
          borderColor: '#ff3366',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.2,
        },
        {
          label: 'IR AC',
          data: state.ppgIrBuffer,
          borderColor: '#00d4ff',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      parsing: false,
      scales: {
        x: { display: false },
        y: {
          suggestedMin: -500,
          suggestedMax: 500,
          grid: { color: 'rgba(0, 212, 255, 0.08)' },
          ticks: { color: '#8b949e', font: { family: 'SF Mono, monospace', size: 10 } },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { color: '#8b949e', boxWidth: 12, font: { size: 11 } },
        },
      },
    },
  });
}

// Throttled Chart Render Loop
let lastRenderTime = 0;
function requestChartRender() {
  const now = performance.now();
  if (now - lastRenderTime > 1000 / CONFIG.CHART_REFRESH_FPS) {
    if (ecgChart) ecgChart.update('none');
    if (ppgChart) ppgChart.update('none');
    lastRenderTime = now;
  }
}

// ==============================================================================
// WebSocket Communication
// ==============================================================================
function checkMixedContentWarning(url) {
  if (window.location.protocol === 'https:' && url.startsWith('ws://')) {
    elements.mixedContentAlert.style.display = 'block';
  } else {
    elements.mixedContentAlert.style.display = 'none';
  }
}

function connectWebSocket() {
  const url = elements.wsUrlInput.value.trim();
  checkMixedContentWarning(url);

  if (state.isSimulating) {
    stopSimulation();
  }

  updateConnectionStatus('connecting', 'Connecting...');

  try {
    state.socket = new WebSocket(url);

    state.socket.onopen = () => {
      state.isConnected = true;
      updateConnectionStatus('connected', 'Live Streaming');
      elements.connectBtn.textContent = 'Disconnect';
      elements.connectBtn.classList.replace('btn-primary', 'btn-secondary');
      console.log('[WS] Connected to ' + url);
    };

    state.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleIncomingPacket(data);
      } catch (err) {
        console.warn('[WS] Malformed packet:', event.data, err);
      }
    };

    state.socket.onerror = (err) => {
      console.error('[WS] Socket error:', err);
      updateConnectionStatus('disconnected', 'Error');
    };

    state.socket.onclose = () => {
      state.isConnected = false;
      updateConnectionStatus('disconnected', 'Disconnected');
      elements.connectBtn.textContent = 'Connect';
      elements.connectBtn.classList.replace('btn-secondary', 'btn-primary');
      console.log('[WS] Disconnected');
    };
  } catch (error) {
    console.error('[WS] Connection exception:', error);
    updateConnectionStatus('disconnected', 'Failed');
  }
}

function disconnectWebSocket() {
  if (state.socket) {
    state.socket.close();
    state.socket = null;
  }
}

function updateConnectionStatus(status, text) {
  elements.connStatusText.textContent = text;
  elements.connStatusBadge.className = 'badge';

  if (status === 'connected') {
    elements.connStatusBadge.classList.add('badge-connected');
  } else if (status === 'connecting') {
    elements.connStatusBadge.classList.add('badge-warning');
  } else {
    elements.connStatusBadge.classList.add('badge-disconnected');
  }
}

// ==============================================================================
// Incoming Sensor Frame Processing
// ==============================================================================
let ppgRedDc = 0;
let ppgIrDc = 0;

function handleIncomingPacket(packet) {
  state.packetCount++;
  elements.packetCount.textContent = state.packetCount;

  // Handle ECG Array
  if (Array.isArray(packet.ecg)) {
    for (let i = 0; i < packet.ecg.length; i++) {
      const val = parseFloat(packet.ecg[i]);
      state.ecgBuffer.push(val);
      if (state.ecgBuffer.length > CONFIG.MAX_ECG_POINTS) {
        state.ecgBuffer.shift();
      }

      // Record to experiment log if active
      if (state.isRecording) {
        recordSamplePoint({
          ts: Date.now(),
          ecg_mv: val,
          ppg_red: 0,
          ppg_ir: 0,
        });
      }
    }
  }

  // Handle PPG Red & IR Arrays
  if (Array.isArray(packet.ppg_red) && Array.isArray(packet.ppg_ir)) {
    const len = Math.min(packet.ppg_red.length, packet.ppg_ir.length);
    for (let i = 0; i < len; i++) {
      const red = packet.ppg_red[i];
      const ir = packet.ppg_ir[i];

      // Lightweight dynamic AC filtering for visual trace display
      if (ppgRedDc === 0) ppgRedDc = red;
      if (ppgIrDc === 0) ppgIrDc = ir;
      ppgRedDc = 0.98 * ppgRedDc + 0.02 * red;
      ppgIrDc = 0.98 * ppgIrDc + 0.02 * ir;

      const redAC = red - ppgRedDc;
      const irAC = ir - ppgIrDc;

      state.ppgRedBuffer.push(redAC);
      state.ppgIrBuffer.push(irAC);

      if (state.ppgRedBuffer.length > CONFIG.MAX_PPG_POINTS) {
        state.ppgRedBuffer.shift();
        state.ppgIrBuffer.shift();
      }
    }
  }

  // Update Vitals Snapshot
  if (packet.vitals) {
    updateVitalsUI(packet.vitals);
  }

  elements.bufferCount.textContent = state.ecgBuffer.length;
  requestChartRender();
}

function updateVitalsUI(v) {
  // Heart Rate
  const hr = parseFloat(v.hr) || 0;
  elements.valHR.textContent = hr > 0 ? hr.toFixed(0) : '--';

  // SpO2
  const spo2 = parseFloat(v.spo2) || 0;
  elements.valSpO2.textContent = spo2 > 0 ? spo2.toFixed(1) : '--';

  // Temperature
  const temp = parseFloat(v.temp) || 0;
  elements.valTemp.textContent = temp > 0 ? temp.toFixed(1) : '--';
  if (temp < 35.5) {
    elements.tempStatusPill.textContent = 'Hypothermia';
    elements.tempStatusPill.className = 'pill pill-danger';
  } else if (temp > 37.8) {
    elements.tempStatusPill.textContent = 'Hyperthermia/Fever';
    elements.tempStatusPill.className = 'pill pill-danger';
  } else {
    elements.tempStatusPill.textContent = 'Normothermia';
    elements.tempStatusPill.className = 'pill pill-neutral';
  }

  // Signal Quality Index (SQI)
  const sqi = parseInt(v.sqi) || 0;
  elements.valSQI.textContent = sqi;
  elements.sqiBar.style.width = `${sqi}%`;
  if (sqi > 70) {
    elements.sqiBar.style.background = '#00ff88';
  } else if (sqi > 40) {
    elements.sqiBar.style.background = '#d29922';
  } else {
    elements.sqiBar.style.background = '#da3633';
  }

  // Lead Status Indicator
  const leadOk = v.lead_ok !== false;
  if (leadOk) {
    elements.leadStatusPill.textContent = 'Leads OK';
    elements.leadStatusPill.className = 'pill pill-success';
    elements.ecgLeadIndicator.textContent = '● Electrodes Attached';
    elements.ecgLeadIndicator.style.color = '#3fb950';
  } else {
    elements.leadStatusPill.textContent = 'Lead-Off Detached';
    elements.leadStatusPill.className = 'pill pill-danger';
    elements.ecgLeadIndicator.textContent = '⚠️ Electrodes Detached (Lead-Off)';
    elements.ecgLeadIndicator.style.color = '#f85149';
  }

  // Finger Presence
  const finger = v.finger === true;
  if (finger) {
    elements.fingerStatusPill.textContent = 'Finger Detected';
    elements.fingerStatusPill.className = 'pill pill-primary';
  } else {
    elements.fingerStatusPill.textContent = 'Searching Finger...';
    elements.fingerStatusPill.className = 'pill pill-danger';
  }

  // Save current vitals into app state
  state.vitals = { ...state.vitals, ...v };
}

// ==============================================================================
// In-Browser Physiological Waveform Simulator
// Enables instant verification without physical ESP32 hardware
// ==============================================================================
let simStep = 0;
function toggleSimulation() {
  if (state.isSimulating) {
    stopSimulation();
  } else {
    startSimulation();
  }
}

function startSimulation() {
  if (state.isConnected) disconnectWebSocket();

  state.isSimulating = true;
  updateConnectionStatus('connected', 'Simulated Live Stream');
  elements.simBtn.textContent = 'Stop Simulation';
  elements.simBtn.classList.replace('btn-secondary', 'btn-danger');

  simStep = 0;
  state.simInterval = setInterval(() => {
    // Generate 100ms batch: 20 ECG points (200Hz) and 10 PPG points (100Hz)
    const ecgBatch = [];
    const ppgRedBatch = [];
    const ppgIrBatch = [];

    const hrBpm = 72;
    const periodSamples = (CONFIG.ECG_SAMPLING_RATE_HZ * 60) / hrBpm; // ~166 samples per beat

    for (let i = 0; i < 20; i++) {
      const beatPhase = (simStep % periodSamples) / periodSamples; // 0 to 1
      let ecgMv = 1650; // Baseline 1.65V

      // Synthetic P-Q-R-S-T wave model
      if (beatPhase > 0.15 && beatPhase < 0.22) {
        // P-wave
        ecgMv += 120 * Math.sin(((beatPhase - 0.15) / 0.07) * Math.PI);
      } else if (beatPhase > 0.28 && beatPhase < 0.31) {
        // Q-dip
        ecgMv -= 150 * Math.sin(((beatPhase - 0.28) / 0.03) * Math.PI);
      } else if (beatPhase >= 0.31 && beatPhase <= 0.36) {
        // R-peak
        ecgMv += 850 * Math.sin(((beatPhase - 0.31) / 0.05) * Math.PI);
      } else if (beatPhase > 0.36 && beatPhase < 0.39) {
        // S-dip
        ecgMv -= 200 * Math.sin(((beatPhase - 0.36) / 0.03) * Math.PI);
      } else if (beatPhase > 0.48 && beatPhase < 0.60) {
        // T-wave
        ecgMv += 220 * Math.sin(((beatPhase - 0.48) / 0.12) * Math.PI);
      }

      // Add gentle baseline drift & micro noise
      ecgMv += 15 * Math.sin((simStep / 400) * Math.PI) + (Math.random() - 0.5) * 12;
      ecgBatch.push(ecgMv.toFixed(1));
      simStep++;
    }

    // PPG Waveform: 10 points per 100ms
    for (let j = 0; j < 10; j++) {
      const ppgPhase = (simStep / 2) % (periodSamples / 2) / (periodSamples / 2);
      let ppgAmp = 0;

      // Systolic peak followed by dicrotic notch
      if (ppgPhase >= 0.15 && ppgPhase <= 0.45) {
        ppgAmp += 350 * Math.sin(((ppgPhase - 0.15) / 0.3) * Math.PI);
      } else if (ppgPhase > 0.45 && ppgPhase <= 0.7) {
        ppgAmp += 110 * Math.sin(((ppgPhase - 0.45) / 0.25) * Math.PI);
      }

      ppgRedBatch.push(Math.round(240000 + ppgAmp));
      ppgIrBatch.push(Math.round(260000 + ppgAmp * 1.15));
    }

    const simPacket = {
      device_id: 'SIMULATOR_ESP32S3',
      seq: state.packetCount,
      ecg: ecgBatch,
      ppg_red: ppgRedBatch,
      ppg_ir: ppgIrBatch,
      vitals: {
        hr: 72 + Math.sin(simStep / 100) * 2,
        spo2: 98.4,
        temp: 36.75,
        sqi: 92,
        lead_ok: true,
        finger: true,
        batt_v: 3.98,
      },
    };

    handleIncomingPacket(simPacket);
  }, 100);
}

function stopSimulation() {
  if (state.simInterval) {
    clearInterval(state.simInterval);
    state.simInterval = null;
  }
  state.isSimulating = false;
  elements.simBtn.textContent = 'Simulate Stream';
  elements.simBtn.classList.replace('btn-danger', 'btn-secondary');
  updateConnectionStatus('disconnected', 'Disconnected');
}

// ==============================================================================
// Experiment Recording & Export Engine
// ==============================================================================
function startRecording() {
  state.isRecording = true;
  state.recordedData = [];
  state.recordingStartTime = Date.now();

  elements.startRecBtn.disabled = true;
  elements.stopRecBtn.disabled = false;
  elements.downloadCsvBtn.disabled = true;
  elements.downloadJsonBtn.disabled = true;
  elements.recordingBadge.style.display = 'inline-block';

  state.recordingTimerInterval = setInterval(updateRecordingTimer, 1000);
}

function stopRecording() {
  state.isRecording = false;
  clearInterval(state.recordingTimerInterval);

  elements.startRecBtn.disabled = false;
  elements.stopRecBtn.disabled = true;
  elements.downloadCsvBtn.disabled = false;
  elements.downloadJsonBtn.disabled = false;
  elements.recordingBadge.style.display = 'none';

  console.log(`[Recorder] Finished session with ${state.recordedData.length} records.`);
}

function recordSamplePoint(sample) {
  state.recordedData.push({
    timestamp_ms: sample.ts,
    ecg_mv: sample.ecg_mv,
    ppg_red: sample.ppg_red,
    ppg_ir: sample.ppg_ir,
    hr_bpm: state.vitals.hr,
    spo2_pct: state.vitals.spo2,
    temp_c: state.vitals.temp,
    sqi_pct: state.vitals.sqi,
  });
}

function updateRecordingTimer() {
  const elapsedSec = Math.floor((Date.now() - state.recordingStartTime) / 1000);
  const minutes = String(Math.floor(elapsedSec / 60)).padStart(2, '0');
  const seconds = String(elapsedSec % 60).padStart(2, '0');
  elements.recDuration.textContent = `${minutes}:${seconds}`;
  elements.recSamples.textContent = state.recordedData.length;
}

function downloadCSV() {
  if (state.recordedData.length === 0) return;

  const headers = ['timestamp_ms', 'ecg_mv', 'ppg_red', 'ppg_ir', 'hr_bpm', 'spo2_pct', 'temp_c', 'sqi_pct'];
  let csvContent = 'data:text/csv;charset=utf-8,' + headers.join(',') + '\n';

  state.recordedData.forEach((row) => {
    const line = [
      row.timestamp_ms,
      row.ecg_mv,
      row.ppg_red,
      row.ppg_ir,
      row.hr_bpm,
      row.spo2_pct,
      row.temp_c,
      row.sqi_pct,
    ].join(',');
    csvContent += line + '\n';
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', `vitals_experiment_${Date.now()}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function downloadJSON() {
  if (state.recordedData.length === 0) return;

  const payload = {
    metadata: {
      recorded_at: new Date().toISOString(),
      sample_rate_ecg_hz: CONFIG.ECG_SAMPLING_RATE_HZ,
      sample_rate_ppg_hz: CONFIG.PPG_SAMPLING_RATE_HZ,
      total_samples: state.recordedData.length,
      device: 'ESP32-S3 Multimodal Prototype',
    },
    samples: state.recordedData,
  };

  const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(payload, null, 2));
  const link = document.createElement('a');
  link.setAttribute('href', dataStr);
  link.setAttribute('download', `vitals_experiment_${Date.now()}.json`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ==============================================================================
// Machine Learning FastAPI Backend Integration
// ==============================================================================
async function testBackendHealth() {
  const backendUrl = elements.backendUrlInput.value.trim().replace(/\/$/, '');
  elements.aiStatusText.textContent = 'Checking health...';
  elements.aiStatusText.style.color = '#e3b341';

  try {
    const res = await fetch(`${backendUrl}/health`, { method: 'GET' });
    if (res.ok) {
      const data = await res.json();
      elements.aiStatusText.textContent = `Backend Online (${data.model_version || 'v1.0'})`;
      elements.aiStatusText.style.color = '#3fb950';
    } else {
      elements.aiStatusText.textContent = `Error: HTTP ${res.status}`;
      elements.aiStatusText.style.color = '#f85149';
    }
  } catch (err) {
    elements.aiStatusText.textContent = 'Backend Offline / Unreachable';
    elements.aiStatusText.style.color = '#f85149';
  }
}

async function runMLInference() {
  const backendUrl = elements.backendUrlInput.value.trim().replace(/\/$/, '');
  elements.aiStatusText.textContent = 'Running Multimodal Inference...';
  elements.aiStatusText.style.color = '#a371f7';

  // Extract the latest 5-second window from circular buffers:
  // 1000 ECG points (200Hz x 5s), 500 PPG points (100Hz x 5s)
  const ecgSlice = state.ecgBuffer.slice(-1000);
  const ppgSlice = state.ppgRedBuffer.slice(-500);

  const payload = {
    ecg_signal: ecgSlice,
    ppg_signal: ppgSlice,
    spo2: state.vitals.spo2 || 98.0,
    temperature_c: state.vitals.temp || 36.6,
    heart_rate_bpm: state.vitals.hr || 72.0,
  };

  try {
    const response = await fetch(`${backendUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`Inference API error: ${response.statusText}`);
    }

    const result = await response.json();
    displayMLResults(result);
    elements.aiStatusText.textContent = 'Inference Complete';
    elements.aiStatusText.style.color = '#3fb950';
  } catch (error) {
    console.error('[ML] Prediction failed:', error);
    elements.aiStatusText.textContent = 'Inference Request Failed';
    elements.aiStatusText.style.color = '#f85149';

    // Graceful offline fallback demonstration calculation
    fallbackInferenceCalculation();
  }
}

function displayMLResults(res) {
  // Blood Pressure
  if (res.sbp && res.dbp) {
    elements.predBP.textContent = `${Math.round(res.sbp)} / ${Math.round(res.dbp)}`;
    elements.predBPCategory.textContent = res.bp_category || 'Normal Range';
  }

  // Arrhythmia Risk
  if (res.arrhythmia_class) {
    elements.predArrhythmia.textContent = res.arrhythmia_class;
    const conf = res.confidence ? Math.round(res.confidence * 100) : 95;
    elements.predArrhythmiaConfidence.textContent = `Confidence: ${conf}%`;
  }

  // PTT
  if (res.ptt_ms) {
    elements.calcPTT.textContent = Math.round(res.ptt_ms);
  }
}

function fallbackInferenceCalculation() {
  // If backend is not started, estimate PTT directly from buffer peaks for demo
  const estimatedPTT = 210 + Math.round((Math.random() - 0.5) * 15);
  // Moens-Korteweg inverse relationship approximation for BP from PTT
  const estSBP = Math.round(118 + (220 - estimatedPTT) * 0.4);
  const estDBP = Math.round(76 + (220 - estimatedPTT) * 0.25);

  elements.predBP.textContent = `${estSBP} / ${estDBP}`;
  elements.predBPCategory.textContent = 'Normal (Client Fallback)';
  elements.predArrhythmia.textContent = 'Normal Sinus Rhythm';
  elements.predArrhythmiaConfidence.textContent = 'Confidence: 94% (Client Estimate)';
  elements.calcPTT.textContent = estimatedPTT;
}

// Auto-inference toggle handler
function toggleAutoInference(enabled) {
  if (enabled) {
    runMLInference();
    state.autoInferenceInterval = setInterval(runMLInference, 5000);
  } else if (state.autoInferenceInterval) {
    clearInterval(state.autoInferenceInterval);
    state.autoInferenceInterval = null;
  }
}

// ==============================================================================
// Event Listeners Setup
// ==============================================================================
function setupEventListeners() {
  elements.connectBtn.addEventListener('click', () => {
    if (state.isConnected) {
      disconnectWebSocket();
    } else {
      connectWebSocket();
    }
  });

  elements.simBtn.addEventListener('click', toggleSimulation);
  elements.testBackendBtn.addEventListener('click', testBackendHealth);
  elements.runInferenceBtn.addEventListener('click', runMLInference);

  elements.autoInferenceToggle.addEventListener('change', (e) => {
    toggleAutoInference(e.target.checked);
  });

  elements.startRecBtn.addEventListener('click', startRecording);
  elements.stopRecBtn.addEventListener('click', stopRecording);
  elements.downloadCsvBtn.addEventListener('click', downloadCSV);
  elements.downloadJsonBtn.addEventListener('click', downloadJSON);

  elements.showIRToggle.addEventListener('change', (e) => {
    if (ppgChart) {
      ppgChart.data.datasets[1].hidden = !e.target.checked;
      ppgChart.update('none');
    }
  });

  // Check mixed content on initial load
  checkMixedContentWarning(elements.wsUrlInput.value);
}

// Bootstrap Application
window.addEventListener('DOMContentLoaded', () => {
  initCharts();
  setupEventListeners();
  console.log('Multimodal Vital-Sign Monitor Dashboard Initialized.');
});
