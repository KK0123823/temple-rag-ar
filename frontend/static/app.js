// frontend/static/app.js
const API_ASK = "/ask";

const qEl = document.getElementById("q");
const btnAsk = document.getElementById("btnAsk");
const btnMic = document.getElementById("btnMic");
const btnReplay = document.getElementById("btnReplay");
const btnPause = document.getElementById("btnPause");

const selEntity = document.getElementById("selEntity");
const selSection = document.getElementById("selSection");
const selTopK = document.getElementById("selTopK");
const chkAutoSpeak = document.getElementById("chkAutoSpeak");
const chkUseLLM = document.getElementById("chkUseLLM");

const hintEl = document.getElementById("hint");
const kidsBox = document.getElementById("kidsBox");
const detailBox = document.getElementById("detailBox");
const citeBox = document.getElementById("citeBox");

const avatarImg = document.getElementById("avatarImg");
const avatarName = document.getElementById("avatarName");

const CHARACTER = {
  mazu:     { name: "媽祖",     avatar: "/static/avatars/mazu.png",     rate: 1.02, pitch: 1.10 },
  wenchang: { name: "文昌帝君", avatar: "/static/avatars/wenchang.png", rate: 1.00, pitch: 1.00 },
  caishen:  { name: "財神爺",   avatar: "/static/avatars/caishen.png",  rate: 1.06, pitch: 0.98 },
  default:  { name: "廟公",     avatar: "/static/avatars/default.png",  rate: 1.00, pitch: 0.95 },
};

let lastKidsText = "";
let lastCharacterKey = "default";
let isSpeakingPaused = false;

// ---------------- UI helpers ----------------
function setHint(msg) {
  hintEl.textContent = msg || "";
}

function setCharacter(key) {
  const c = CHARACTER[key] || CHARACTER.default;
  avatarImg.src = c.avatar;
  avatarName.textContent = c.name;
  lastCharacterKey = key;
}

function renderCitations(citations) {
  if (!citations || citations.length === 0) {
    citeBox.textContent = "(無引用資料)";
    return;
  }
  citeBox.textContent = citations.join("\n");
}

// ---------------- Speech output ----------------
function stopSpeaking() {
  if ("speechSynthesis" in window) {
    window.speechSynthesis.cancel();
  }
  isSpeakingPaused = false;
}

function speak(text, characterKey) {
  if (!("speechSynthesis" in window)) {
    setHint("⚠️ 你的瀏覽器不支援語音輸出（speechSynthesis）");
    return;
  }
  if (!text || !text.trim()) return;

  stopSpeaking();

  const u = new SpeechSynthesisUtterance(text);
  const c = CHARACTER[characterKey] || CHARACTER.default;
  u.lang = "zh-TW";
  u.rate = c.rate;
  u.pitch = c.pitch;

  u.onend = () => { isSpeakingPaused = false; };
  u.onerror = () => { isSpeakingPaused = false; };

  window.speechSynthesis.speak(u);
}

function togglePauseResume() {
  if (!("speechSynthesis" in window)) return;

  // 如果沒有正在朗讀，就當作「再播一次」
  if (!window.speechSynthesis.speaking && lastKidsText) {
    speak(lastKidsText, lastCharacterKey);
    return;
  }

  if (window.speechSynthesis.paused) {
    window.speechSynthesis.resume();
    isSpeakingPaused = false;
  } else {
    window.speechSynthesis.pause();
    isSpeakingPaused = true;
  }
}

// ---------------- Speech input ----------------
let recognition = null;
let recognizing = false;

function setupSpeechInput() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) {
    btnMic.disabled = true;
    btnMic.title = "你的瀏覽器不支援語音輸入";
    return;
  }

  recognition = new SR();
  recognition.lang = "zh-TW";
  recognition.continuous = false;
  recognition.interimResults = false;

  recognition.onstart = () => {
    recognizing = true;
    setHint("🎤 正在聽你說話…");
  };

  recognition.onend = () => {
    recognizing = false;
  };

  recognition.onerror = () => {
    recognizing = false;
    setHint("⚠️ 語音輸入失敗，請再試一次");
  };

  recognition.onresult = (e) => {
    const t = e.results?.[0]?.[0]?.transcript || "";
    if (t.trim()) {
      qEl.value = t.trim();
      setHint("✅ 語音輸入完成！可以直接按「提問」。");
    }
  };
}

function startSpeechInput() {
  if (!recognition) return;
  if (recognizing) {
    recognition.stop();
    recognizing = false;
    return;
  }
  recognition.start();
}

// ---------------- API ----------------
async function ask() {
  const query = (qEl.value || "").trim();
  if (!query) {
    setHint("請先輸入問題喔～");
    return;
  }

  // ✅ 新提問時：先停止上一段語音（避免一直播）
  stopSpeaking();

  setHint("思考中…");
  kidsBox.textContent = "（思考中…）";
  detailBox.textContent = "（思考中…）";
  citeBox.textContent = "";

  const payload = {
    query,
    top_k: parseInt(selTopK.value || "4", 10),
    entity: selEntity.value || null,
    section: selSection.value || null,
    use_llm: !!chkUseLLM.checked,
  };

  let data;
  try {
    const r = await fetch(API_ASK, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    data = await r.json();
  } catch (e) {
    setHint("❌ 連線失敗：請確認後端有開（uvicorn）");
    kidsBox.textContent = "（連線失敗）";
    detailBox.textContent = "（連線失敗）";
    return;
  }

  if (!data || data.ok !== true) {
    setHint("❌ 回傳格式錯誤");
    kidsBox.textContent = "（發生錯誤）";
    detailBox.textContent = "（發生錯誤）";
    return;
  }

  const character = data.character || "default";
  setCharacter(character);

  const kids = data.kids || "";
  const detail = data.detail || "";
  const citations = data.citations || [];

  kidsBox.textContent = kids || "（沒有回答）";
  detailBox.textContent = detail || "（沒有回答）";
  renderCitations(citations);

  lastKidsText = kids;

  setHint("");

  // ✅ 自動朗讀：只播 kids
  if (chkAutoSpeak.checked && kids && kids.trim()) {
    speak(kids, character);
  }
}

// ---------------- events ----------------
btnAsk.addEventListener("click", ask);
qEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});

btnMic.addEventListener("click", startSpeechInput);

btnReplay.addEventListener("click", () => {
  if (!lastKidsText) return;
  speak(lastKidsText, lastCharacterKey);
});

btnPause.addEventListener("click", () => {
  togglePauseResume();
});

setupSpeechInput();
