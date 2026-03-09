const STORAGE_KEY = "ii-lib-auth";

const state = {
  activeUser: null,
  activeSessionId: null,
};

const el = {
  authScreen: document.getElementById("auth-screen"),
  appScreen: document.getElementById("app-screen"),
  loginForm: document.getElementById("login-form"),
  loginName: document.getElementById("login-name"),
  authError: document.getElementById("auth-error"),
  currentUser: document.getElementById("current-user"),
  logoutBtn: document.getElementById("logout-btn"),
  uploadForm: document.getElementById("upload-form"),
  docsList: document.getElementById("documents-list"),
  refreshDocsBtn: document.getElementById("refresh-docs"),
  chatLog: document.getElementById("chat-log"),
  chatForm: document.getElementById("chat-form"),
  chatMessage: document.getElementById("chat-message"),
  newChatBtn: document.getElementById("new-chat-btn"),
  loadHistoryBtn: document.getElementById("load-history-btn"),
  statusLog: document.getElementById("status-log"),
};

function showAuthError(message) {
  if (!message) {
    el.authError.textContent = "";
    el.authError.classList.add("hidden");
    return;
  }
  el.authError.textContent = message;
  el.authError.classList.remove("hidden");
}

function logStatus(message, level = "info") {
  const line = `[${new Date().toLocaleTimeString()}] ${level.toUpperCase()} ${message}`;
  el.statusLog.textContent = `${line}\n${el.statusLog.textContent}`.trim();
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try {
      const data = await response.json();
      if (data?.detail) {
        message = data.detail;
      }
    } catch (_err) {
      // Ignore non-JSON error body parsing issues.
    }
    throw new Error(message);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

function appendChatMessage(role, content) {
  const msg = document.createElement("div");
  msg.className = `msg ${role}`;
  msg.textContent = content;
  el.chatLog.append(msg);
  el.chatLog.scrollTop = el.chatLog.scrollHeight;
  return msg;
}

function setAuthUi(isLoggedIn) {
  el.authScreen.classList.toggle("hidden", isLoggedIn);
  el.appScreen.classList.toggle("hidden", !isLoggedIn);
}

function persistAuth() {
  if (!state.activeUser) {
    localStorage.removeItem(STORAGE_KEY);
    return;
  }
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      userId: state.activeUser.id,
      userName: state.activeUser.name,
      sessionId: state.activeSessionId,
    }),
  );
}

function loadPersistedAuth() {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw);
    if (!parsed?.userId || !parsed?.userName) {
      return null;
    }
    return parsed;
  } catch (_err) {
    return null;
  }
}

function renderCurrentUser() {
  if (!state.activeUser) {
    el.currentUser.textContent = "";
    return;
  }
  el.currentUser.textContent = `Вы вошли как: ${state.activeUser.name} (#${state.activeUser.id})`;
}

async function findOrCreateUserByName(name) {
  const trimmed = name.trim();
  if (!trimmed) {
    throw new Error("Введите имя пользователя");
  }

  // First try exact-ish lookup to avoid duplicate users.
  try {
    const users = await apiFetch(`/users/?name=${encodeURIComponent(trimmed)}&limit=100&offset=0`);
    const existing = users.find((user) => user.name.toLowerCase() === trimmed.toLowerCase());
    if (existing) {
      return existing;
    }
  } catch (_err) {
    // Continue with create fallback if search endpoint is temporarily unavailable.
  }

  try {
    return await apiFetch("/users/", {
      method: "POST",
      body: JSON.stringify({ name: trimmed }),
    });
  } catch (err) {
    // Handle race/duplicate: if create failed, fetch and reuse existing user.
    const users = await apiFetch(`/users/?name=${encodeURIComponent(trimmed)}&limit=100&offset=0`);
    const existing = users.find((user) => user.name.toLowerCase() === trimmed.toLowerCase());
    if (existing) {
      return existing;
    }
    throw err;
  }
}

function renderDocuments(docs) {
  el.docsList.innerHTML = "";

  if (!docs.length) {
    const li = document.createElement("li");
    li.textContent = "Документов пока нет";
    el.docsList.append(li);
    return;
  }

  docs.forEach((doc) => {
    const li = document.createElement("li");
    const left = document.createElement("div");
    const title = document.createElement("div");
    const meta = document.createElement("div");
    const removeBtn = document.createElement("button");

    title.textContent = doc.filename;
    meta.className = "meta";
    meta.textContent = `${doc.status} | ${doc.file_size} bytes`;
    left.append(title, meta);

    removeBtn.className = "danger";
    removeBtn.textContent = "Удалить";
    removeBtn.addEventListener("click", async () => {
      try {
        await apiFetch(`/documents/${doc.id}`, { method: "DELETE" });
        logStatus(`Документ удален: ${doc.filename}`);
        await loadDocuments();
      } catch (err) {
        logStatus(`Ошибка удаления документа: ${err.message}`, "error");
      }
    });

    li.append(left, removeBtn);
    el.docsList.append(li);
  });
}

async function loadDocuments() {
  if (!state.activeUser) {
    return;
  }
  const docs = await apiFetch(`/documents/?user_id=${state.activeUser.id}`);
  renderDocuments(docs);
}

async function fetchSessions() {
  if (!state.activeUser) {
    return [];
  }
  return apiFetch(`/chat/sessions?user_id=${state.activeUser.id}`);
}

async function loadLatestSessionHistory() {
  const sessions = await fetchSessions();
  if (!sessions.length) {
    state.activeSessionId = null;
    persistAuth();
    el.chatLog.innerHTML = "";
    logStatus("История диалогов пока пустая");
    return;
  }

  state.activeSessionId = sessions[0].id;
  persistAuth();

  const messages = await apiFetch(`/chat/sessions/${state.activeSessionId}/messages`);
  el.chatLog.innerHTML = "";
  messages.forEach((msg) => appendChatMessage(msg.role, msg.content));
  logStatus(`Открыт последний диалог (${messages.length} сообщений)`);
}

async function restoreSessionOrFallback() {
  const sessions = await fetchSessions();
  if (!sessions.length) {
    state.activeSessionId = null;
    persistAuth();
    return;
  }

  const hasPersisted = state.activeSessionId && sessions.some((s) => s.id === state.activeSessionId);
  state.activeSessionId = hasPersisted ? state.activeSessionId : sessions[0].id;
  persistAuth();
}

async function sendChatMessage(event) {
  event.preventDefault();

  if (!state.activeUser) {
    logStatus("Сначала выполните вход", "error");
    return;
  }

  const content = el.chatMessage.value.trim();
  if (!content) {
    return;
  }

  appendChatMessage("user", content);
  el.chatMessage.value = "";
  const assistantEl = appendChatMessage("assistant", "");

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: state.activeUser.id,
        session_id: state.activeSessionId,
        message: content,
      }),
    });

    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let pendingEvent = "message";

    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() || "";

      for (const chunk of chunks) {
        const lines = chunk.split("\n");
        const dataLines = [];

        for (const line of lines) {
          if (line.startsWith("event:")) {
            pendingEvent = line.slice(6).trim();
          }
          if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trimStart());
          }
        }

        const payload = dataLines.join("\n");
        if (!payload) {
          continue;
        }

        if (pendingEvent === "token" || pendingEvent === "message") {
          assistantEl.textContent += payload;
        } else if (pendingEvent === "error") {
          assistantEl.textContent += `\n[error] ${payload}`;
          logStatus(`Ошибка SSE: ${payload}`, "error");
        } else {
          logStatus(`${pendingEvent}: ${payload}`);
        }
      }

      el.chatLog.scrollTop = el.chatLog.scrollHeight;
    }

    // Session may be created by backend on first message; detect it by reloading sessions.
    await restoreSessionOrFallback();
    logStatus("Ответ готов");
  } catch (err) {
    assistantEl.textContent += `\n[client error] ${err.message}`;
    logStatus(`Ошибка чата: ${err.message}`, "error");
  }
}

async function signIn(user) {
  state.activeUser = user;
  renderCurrentUser();
  showAuthError("");
  setAuthUi(true);
  const preloadResults = await Promise.allSettled([loadDocuments(), restoreSessionOrFallback()]);
  if (preloadResults[0].status === "rejected") {
    logStatus(`Не удалось загрузить документы: ${preloadResults[0].reason?.message || "unknown error"}`, "error");
  }
  if (preloadResults[1].status === "rejected") {
    logStatus(`Не удалось восстановить диалог: ${preloadResults[1].reason?.message || "unknown error"}`, "error");
  }
  persistAuth();
  logStatus(`Вход выполнен: ${user.name}`);
  el.chatMessage.focus();
}

function signOut() {
  state.activeUser = null;
  state.activeSessionId = null;
  persistAuth();
  el.chatLog.innerHTML = "";
  el.docsList.innerHTML = "";
  renderCurrentUser();
  showAuthError("");
  setAuthUi(false);
  logStatus("Сессия завершена");
}

el.loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = String(new FormData(el.loginForm).get("name") || "").trim();
  showAuthError("");

  try {
    const user = await findOrCreateUserByName(name);
    await signIn(user);
    el.loginForm.reset();
  } catch (err) {
    showAuthError(`Не удалось войти: ${err.message}`);
    logStatus(`Ошибка входа: ${err.message}`, "error");
  }
});

el.logoutBtn.addEventListener("click", signOut);

el.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!state.activeUser) {
    logStatus("Сначала выполните вход", "error");
    return;
  }

  const form = new FormData(el.uploadForm);
  const file = form.get("file");
  if (!(file instanceof File) || !file.size) {
    logStatus("Выберите файл для загрузки", "error");
    return;
  }

  const body = new FormData();
  body.append("file", file);

  try {
    const response = await fetch(`/documents/upload?user_id=${state.activeUser.id}`, {
      method: "POST",
      body,
    });

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        detail = data?.detail || detail;
      } catch (_err) {
        // Ignore non-JSON error body parsing issues.
      }
      throw new Error(detail);
    }

    const doc = await response.json();
    el.uploadForm.reset();
    await loadDocuments();
    logStatus(`Документ загружен: ${doc.filename}`);
  } catch (err) {
    logStatus(`Ошибка загрузки документа: ${err.message}`, "error");
  }
});

el.refreshDocsBtn.addEventListener("click", async () => {
  try {
    await loadDocuments();
    logStatus("Список документов обновлен");
  } catch (err) {
    logStatus(`Ошибка обновления документов: ${err.message}`, "error");
  }
});

el.newChatBtn.addEventListener("click", () => {
  state.activeSessionId = null;
  persistAuth();
  el.chatLog.innerHTML = "";
  logStatus("Создан новый контекст диалога");
});

el.loadHistoryBtn.addEventListener("click", async () => {
  try {
    await loadLatestSessionHistory();
  } catch (err) {
    logStatus(`Ошибка загрузки истории: ${err.message}`, "error");
  }
});

el.chatForm.addEventListener("submit", sendChatMessage);

async function bootstrap() {
  const saved = loadPersistedAuth();
  if (!saved) {
    setAuthUi(false);
    return;
  }

  try {
    const user = await apiFetch(`/users/${saved.userId}`);
    state.activeSessionId = saved.sessionId || null;
    await signIn(user);
  } catch (_err) {
    signOut();
  }
}

bootstrap();
