const input = document.querySelector('#media-input');
const selection = document.querySelector('#selection');
const uploadButton = document.querySelector('#upload-button');
const jobs = document.querySelector('#jobs');
const refreshButton = document.querySelector('#refresh-button');
const filter = document.querySelector('#filter');
const search = document.querySelector('#search');
const historyActions = document.querySelector('#history-actions');
const themeToggle = document.querySelector('#theme-toggle');
const soundToggle = document.querySelector('#sound-toggle');
const appLoading = document.querySelector('#app-loading');
let selectedFiles = [];
let allJobs = [];
let lastJobsSignature = '';
let audioContext;
let jobsRequest = null;
let pollingTimer = null;

function finishAppLoading() {
  if (!appLoading) return;
  appLoading.classList.add('ready');
  setTimeout(() => appLoading.remove(), 360);
}

function applyTheme(retro) {
  document.body.classList.toggle('retro', retro);
  themeToggle.textContent = retro ? 'Normales Design' : 'Retro testen';
  themeToggle.setAttribute('aria-pressed', String(retro));
}

applyTheme(localStorage.getItem('ig-automatik-theme') === 'retro');
themeToggle.addEventListener('click', () => {
  const retro = !document.body.classList.contains('retro');
  localStorage.setItem('ig-automatik-theme', retro ? 'retro' : 'normal');
  applyTheme(retro);
  playSound('click');
});

function applySound(enabled) {
  soundToggle.textContent = enabled ? 'Sound: an' : 'Sound: aus';
  soundToggle.setAttribute('aria-pressed', String(enabled));
}

applySound(localStorage.getItem('ig-automatik-sound') === 'on');
soundToggle.addEventListener('click', () => {
  const enabled = localStorage.getItem('ig-automatik-sound') !== 'on';
  localStorage.setItem('ig-automatik-sound', enabled ? 'on' : 'off');
  applySound(enabled);
  if (enabled) playSound('success');
});

function playSound(kind = 'click') {
  if (localStorage.getItem('ig-automatik-sound') !== 'on') return;
  try {
    audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === 'suspended') audioContext.resume();
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    const now = audioContext.currentTime;
    oscillator.type = 'square';
    oscillator.frequency.setValueAtTime(kind === 'success' ? 660 : 440, now);
    oscillator.frequency.exponentialRampToValueAtTime(kind === 'success' ? 880 : 330, now + 0.07);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.035, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.09);
    oscillator.connect(gain).connect(audioContext.destination);
    oscillator.start(now);
    oscillator.stop(now + 0.1);
  } catch {
    // Sound is optional; the app must continue normally if audio is blocked.
  }
}

input.addEventListener('change', () => {
  selectedFiles = Array.from(input.files || []);
  selection.textContent = selectedFiles.length
    ? selectedFiles.map(file => `${file.name} (${formatBytes(file.size)})`).join(', ')
    : '';
  uploadButton.disabled = selectedFiles.length === 0;
});

uploadButton.addEventListener('click', async () => {
  playSound('click');
  uploadButton.disabled = true;
  try {
    for (let index = 0; index < selectedFiles.length; index++) {
      const file = selectedFiles[index];
      uploadButton.textContent = `Upload ${index + 1}/${selectedFiles.length} …`;
      await upload(file, (percent) => {
        uploadButton.textContent = `Upload ${index + 1}/${selectedFiles.length} · ${percent}%`;
      });
    }
    selectedFiles = [];
    input.value = '';
    selection.textContent = 'Upload abgeschlossen. Verarbeitung läuft.';
    await loadJobs();
    playSound('success');
  } catch (error) {
    alert(error.message || 'Upload fehlgeschlagen.');
  } finally {
    uploadButton.disabled = selectedFiles.length === 0;
    uploadButton.textContent = 'Verarbeitung starten';
  }
});

refreshButton.addEventListener('click', () => { playSound('click'); loadJobs(true); });
filter.addEventListener('change', () => renderJobs(allJobs));
search.addEventListener('input', () => renderJobs(allJobs));

function upload(file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', '/api/upload');
    request.setRequestHeader('Content-Type', 'application/octet-stream');
    request.setRequestHeader('X-Filename', encodeURIComponent(file.name));
    request.upload.onprogress = event => {
      if (event.lengthComputable && onProgress) onProgress(Math.round(event.loaded / event.total * 100));
    };
    request.onload = () => {
      if (request.status >= 200 && request.status < 300) resolve();
      else reject(new Error(readError(request.responseText)));
    };
    request.onerror = () => reject(new Error('Der Server ist nicht erreichbar.'));
    request.send(file);
  });
}

async function loadJobs(force = false) {
  if (!force && document.visibilityState !== 'visible') return;
  if (jobsRequest) return jobsRequest;

  jobsRequest = (async () => {
    try {
      const response = await fetch('/api/jobs', { cache: 'no-store' });
      if (!response.ok) throw new Error('Status konnte nicht geladen werden.');
      const data = await response.json();
      allJobs = data.jobs || [];
      // Do not rebuild image/video cards when nothing changed. Replacing the
      // cards every few seconds makes Safari visibly reload the previews.
      const signature = JSON.stringify(allJobs);
      if (force || signature !== lastJobsSignature) {
        lastJobsSignature = signature;
        renderJobs(allJobs);
      }
    } catch (error) {
      jobs.innerHTML = `<div class="empty error">${escapeHtml(error.message)}</div>`;
    }
  })();

  try {
    await jobsRequest;
  } finally {
    jobsRequest = null;
  }
}

function renderJobs(items) {
  const selectedFilter = filter.value;
  const searchTerm = search.value.trim().toLowerCase();
  items = selectedFilter === 'all'
    ? items
    : items.filter(job => Object.prototype.hasOwnProperty.call(job.outputs || {}, selectedFilter));
  if (searchTerm) {
    items = items.filter(job => String(job.original_name || '').toLowerCase().includes(searchTerm));
  }
  const totalFiltered = items.length;
  if (!items.length) {
    jobs.innerHTML = `<div class="empty">${searchTerm ? 'Keine passenden Dateien.' : selectedFilter === 'all' ? 'Noch keine Uploads.' : `Noch keine ${selectedFilter}-Ergebnisse.`}</div>`;
    historyActions.innerHTML = '';
    return;
  }
  // No artificial history limit: every job returned by the local bridge stays
  // accessible. Search and format filters keep a long history usable.
  jobs.innerHTML = items.map(job => {
    const formats = selectedFilter === 'all' ? Object.entries(job.outputs || {}) : [[selectedFilter, job.outputs[selectedFilter]]];
    const outputGroups = formats.map(([format, files]) => `
      <div class="format-block">
        <div class="format-title">${formatLabel(format)}</div>
        <div class="variants">${files.map(file => variantCard(file)).join('')}</div>
      </div>`).join('');
    const statusText = {
      done: 'Fertig',
      processing: 'Wird verarbeitet',
      missing_outputs: 'Abgeschlossen – Dateien fehlen',
      not_received: 'Upload nicht angekommen',
    }[job.status] || job.status;
    const statusNote = {
      missing_outputs: 'Manifest vorhanden, aber die fertigen Dateien fehlen.',
      not_received: 'Kein Original und kein Abschluss-Manifest gefunden.',
    }[job.status];
    const canReprocess = job.status === 'done' || job.status === 'missing_outputs';
    const historyButtons = `
      <div class="job-actions">
        ${canReprocess ? `<button class="reprocess" data-job-id="${escapeHtml(job.id)}">Erneut verarbeiten</button>` : ''}
        <button class="history-remove" data-job-id="${escapeHtml(job.id)}">Eintrag entfernen</button>
      </div>`;
    const pendingView = job.status === 'processing'
      ? '<div class="progress"><span></span></div>'
      : statusNote ? `<div class="status-note">${statusNote}</div>` : '';
    return `<article class="job">
    <div class="job-top"><div><strong>${escapeHtml(job.original_name)}</strong><small>${formatDate(job.created)}</small></div><span class="status ${job.status}">${statusText}</span></div>
      ${outputGroups || pendingView}
      ${historyButtons}
    </article>`;
  }).join('');
  bindShareButtons();
  bindHistoryButtons();
  historyActions.innerHTML = `<small class="history-note">${totalFiltered} ${totalFiltered === 1 ? 'Ergebnis' : 'Ergebnisse'} angezeigt</small>`;
}

function bindHistoryButtons() {
  document.querySelectorAll('.history-remove').forEach(button => {
    button.addEventListener('click', async () => {
      if (!confirm('Nur diesen Eintrag aus der App-Historie entfernen? Pipeline-Dateien bleiben erhalten.')) return;
      button.disabled = true;
      try {
        const response = await fetch(`/api/jobs/${encodeURIComponent(button.dataset.jobId)}`, { method: 'DELETE' });
        if (!response.ok) throw new Error(readError(await response.text(), 'Eintrag konnte nicht entfernt werden.'));
        await loadJobs(true);
      } catch (error) {
        button.disabled = false;
        alert(error.message || 'Eintrag konnte nicht entfernt werden.');
      }
    });
  });
  document.querySelectorAll('.reprocess').forEach(button => {
    button.addEventListener('click', async () => {
      button.disabled = true;
      const originalText = button.textContent;
      button.textContent = 'Wird vorbereitet …';
      try {
        const response = await fetch(`/api/jobs/${encodeURIComponent(button.dataset.jobId)}/reprocess`, { method: 'POST' });
        if (!response.ok) throw new Error(readError(await response.text(), 'Erneute Verarbeitung konnte nicht gestartet werden.'));
        await loadJobs(true);
      } catch (error) {
        button.disabled = false;
        button.textContent = originalText;
        alert(error.message || 'Erneute Verarbeitung konnte nicht gestartet werden.');
      }
    });
  });
}

function variantCard(file) {
  const video = /\.(mp4|mov|avi|mkv|webm|m4v|3gp)$/i.test(file.name);
  const preview = video
    ? `<video class="preview" src="${file.preview_url}" poster="${file.poster_url || ''}" controls preload="metadata" playsinline></video>`
    : `<img class="preview" src="${file.preview_url}" alt="Variante ${file.variant}" loading="lazy">`;
  return `<div class="variant-card">
    <div class="variant-heading"><strong>Variante ${file.variant}</strong><span>${file.variant === 'A' ? 'Natural' : 'Cinematic'}</span></div>
    ${preview}
    <div class="card-actions">
      <button class="share" data-action="save" data-url="${file.url}" data-name="${escapeHtml(file.name)}" data-video="${video}">In Fotos speichern</button>
      <button class="share ${file.variant === 'B' ? 'accent' : ''}" data-action="share" data-url="${file.url}" data-name="${escapeHtml(file.name)}" data-video="${video}">↗ Teilen</button>
    </div>
    ${file.master_url ? `<div class="master-actions"><button class="share master-share" data-action="save" data-url="${file.master_url}" data-name="${escapeHtml(file.master_name)}" data-video="${video}">Vollqualität in Fotos speichern</button></div>` : `<div class="master-missing">Vollqualität derzeit nicht im Archiv gefunden</div>`}
  </div>`;
}

function formatLabel(format) {
  return { POSTS: 'Instagram-Post · 4:5', STORIES: 'Story · 9:16', REELS: 'Reel · 9:16' }[format] || format;
}

let nativeMediaPlugin;

function getNativeMediaPlugin() {
  const capacitor = window.Capacitor;
  if (!capacitor) return null;
  if (typeof capacitor.isNativePlatform === 'function' && !capacitor.isNativePlatform()) return null;
  if (capacitor.Plugins?.IGMedia) return capacitor.Plugins.IGMedia;
  if (typeof capacitor.registerPlugin === 'function') {
    nativeMediaPlugin ||= capacitor.registerPlugin('IGMedia');
    return nativeMediaPlugin;
  }
  return null;
}

function showActionToast(message, error = false) {
  let toast = document.getElementById('action-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'action-toast';
    toast.setAttribute('role', 'status');
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.toggle('error', error);
  toast.classList.add('visible');
  clearTimeout(showActionToast.timer);
  showActionToast.timer = setTimeout(() => toast.classList.remove('visible'), 2600);
}

async function runNativeMediaAction(action, url, filename, video) {
  const plugin = getNativeMediaPlugin();
  if (!plugin || typeof plugin[action] !== 'function') return false;
  const absoluteURL = new URL(url, window.location.href).toString();
  await plugin[action]({
    url: absoluteURL,
    filename: filename || (video ? 'IG-AUTOMATIK.mp4' : 'IG-AUTOMATIK.jpg'),
    kind: video ? 'video' : 'image',
  });
  if (action === 'saveToPhotos') showActionToast('✓ In Fotos gespeichert');
  return true;
}

async function shareOutput(url, filename, video = false, trigger = null, action = 'share') {
  if (trigger?.dataset.busy === 'true') return;
  const originalText = trigger?.textContent;
  if (trigger) {
    trigger.dataset.busy = 'true';
    trigger.setAttribute('aria-busy', 'true');
    trigger.classList.add('busy');
    if ('disabled' in trigger) trigger.disabled = true;
    trigger.textContent = 'Wird vorbereitet …';
  }
  try {
    const nativeAction = action === 'save' ? 'saveToPhotos' : 'shareFile';
    if (await runNativeMediaAction(nativeAction, url, filename, video)) return;

    const response = await fetch(url);
    if (!response.ok) throw new Error('Datei konnte nicht geladen werden.');
    const blob = await response.blob();
    const file = new File([blob], filename, { type: blob.type || 'application/octet-stream' });
    let canShareFile = Boolean(navigator.share);
    if (canShareFile && navigator.canShare) {
      try {
        canShareFile = navigator.canShare({ files: [file] });
      } catch {
        canShareFile = false;
      }
    }
    if (canShareFile) {
      try {
        await navigator.share({ title: 'IG-AUTOMATIK', files: [file] });
        return;
      } catch (error) {
        if (error.name === 'AbortError') return;
        openMediaViewer({
          downloadUrl: url,
          previewUrl: url,
          filename,
          video,
          fallbackMessage: video
            ? 'Das iPhone-Teilen konnte nicht gestartet werden. Bitte öffne den Viewer erneut und versuche „In Fotos sichern / teilen“ noch einmal.'
            : 'Das iPhone-Teilen konnte nicht gestartet werden. Halte das Bild direkt gedrückt und wähle „In Fotos sichern“.',
        });
        return;
      }
    }
    openMediaViewer({
      downloadUrl: url,
      previewUrl: url,
      filename,
      video,
      fallbackMessage: video
        ? 'Das iPhone-Teilen ist hier nicht verfügbar. Bitte versuche „In Fotos sichern / teilen“ noch einmal.'
        : 'Das iPhone-Teilen ist hier nicht verfügbar. Halte das Bild direkt gedrückt und wähle „In Fotos sichern“.',
    });
  } catch (error) {
    if (error.name !== 'AbortError') alert(error.message || 'Sichern fehlgeschlagen.');
  } finally {
    if (trigger) {
      delete trigger.dataset.busy;
      trigger.removeAttribute('aria-busy');
      trigger.classList.remove('busy');
      if ('disabled' in trigger) trigger.disabled = false;
      trigger.textContent = originalText;
    }
  }
}

let mediaViewer;

function inlineUrl(url) {
  return `${url}${url.includes('?') ? '&' : '?'}inline=1`;
}

function isAppleMobile() {
  return /iPad|iPhone|iPod/.test(navigator.userAgent)
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}

function createMediaViewer() {
  const root = document.createElement('div');
  root.className = 'media-viewer';
  root.hidden = true;
  root.innerHTML = `
    <div class="viewer-panel" role="dialog" aria-modal="true" aria-label="Datei ansehen">
      <div class="viewer-top"><strong class="viewer-title"></strong><button class="viewer-close" type="button" aria-label="Schließen">×</button></div>
      <div class="viewer-media"></div>
      <div class="viewer-actions">
        <button class="share viewer-share" type="button">In Fotos sichern</button>
        <a class="download viewer-download">Download am Computer</a>
      </div>
      <p class="viewer-hint">Auf dem iPhone bleibt die App geöffnet. Nutze „In Fotos sichern / teilen“.</p>
    </div>`;
  document.body.appendChild(root);
  const close = () => {
    root.hidden = true;
    document.body.classList.remove('viewer-open');
    root.querySelector('.viewer-media').replaceChildren();
  };
  root.querySelector('.viewer-close').addEventListener('click', close);
  root.addEventListener('click', event => { if (event.target === root) close(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !root.hidden) close(); });
  return { root, close };
}

function openMediaViewer({ downloadUrl, previewUrl, filename, video = false, fallbackMessage = '' }) {
  if (!mediaViewer) mediaViewer = createMediaViewer();
  const { root } = mediaViewer;
  const media = root.querySelector('.viewer-media');
  const safeName = filename || 'Datei';
  root.querySelector('.viewer-title').textContent = safeName;
  media.replaceChildren();
  const element = document.createElement(video ? 'video' : 'img');
  element.className = 'viewer-preview';
  element.src = inlineUrl(previewUrl || downloadUrl);
  if (video) {
    element.controls = true;
    element.preload = 'metadata';
    element.playsInline = true;
  } else {
    element.alt = safeName;
  }
  media.appendChild(element);
  const download = root.querySelector('.viewer-download');
  download.href = downloadUrl;
  download.download = safeName;
  download.hidden = isAppleMobile();
  const share = root.querySelector('.viewer-share');
  share.textContent = isAppleMobile() ? 'In Fotos sichern / teilen' : 'In Fotos sichern';
  root.querySelector('.viewer-hint').textContent = fallbackMessage || (isAppleMobile()
    ? 'Auf dem iPhone bleibt die App geöffnet. Nutze „In Fotos sichern / teilen“.'
    : 'Über „Download am Computer“ wird die Datei direkt heruntergeladen.');
  share.onclick = () => shareOutput(downloadUrl, safeName, video, share, 'save');
  root.hidden = false;
  document.body.classList.add('viewer-open');
}

function bindShareButtons() {
  document.querySelectorAll('.share').forEach(button => {
    button.addEventListener('click', () => {
      playSound('click');
      shareOutput(
        button.dataset.url,
        button.dataset.name,
        button.dataset.video === 'true',
        button,
        button.dataset.action || 'save',
      );
    });
  });
  document.querySelectorAll('[data-download="true"]').forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      playSound('click');
      // On iPhone this opens the native share sheet, where the user can
      // choose “Video sichern” instead of being sent to the file viewer.
      shareOutput(
        link.dataset.url || link.href,
        link.dataset.name || 'Datei',
        link.dataset.video === 'true',
        link,
        'share',
      );
    });
  });
}

function formatBytes(value) {
  if (value < 1024 * 1024) return `${Math.round(value / 1024)} KB`;
  return `${(value / 1024 / 1024).toFixed(1)} MB`;
}
function formatDate(value) {
  try { return new Date(value).toLocaleString('de-DE', { dateStyle: 'short', timeStyle: 'short' }); }
  catch { return ''; }
}
function readError(text, fallback = 'Upload fehlgeschlagen.') {
  try { return JSON.parse(text).error || fallback; }
  catch { return fallback; }
}
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function schedulePolling() {
  clearTimeout(pollingTimer);
  pollingTimer = null;
  if (document.visibilityState !== 'visible') return;
  pollingTimer = setTimeout(async () => {
    await loadJobs();
    schedulePolling();
  }, 3000);
}

loadJobs(true).finally(() => {
  finishAppLoading();
  schedulePolling();
});

// iOS may suspend timers while the Home-Screen app is in the background. A
// fresh request when the app becomes visible prevents a stale "Wartet auf
// Verarbeitung" status after the main pipeline has already finished.
let lastResumeRefresh = 0;
function refreshOnResume() {
  const now = Date.now();
  if (now - lastResumeRefresh < 1000) return;
  lastResumeRefresh = now;
  loadJobs(true).finally(schedulePolling);
}

window.addEventListener('pageshow', refreshOnResume);
window.addEventListener('focus', refreshOnResume);
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'visible') refreshOnResume();
  else schedulePolling();
});
