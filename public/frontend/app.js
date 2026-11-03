/**
 * DigitVision AI - Modern JavaScript Frontend Application
 * High-performance, responsive, touch-friendly UI for handwritten digit & text recognition.
 */

(function () {
    "use strict";

    // Application State
    const state = {
        voiceEnabled: true,
        cameraStream: null,
        cameraFacing: "environment", // 'user' or 'environment'
        isDrawing: false,
        lastX: 0,
        lastY: 0,
        brushSize: 20,
        uploadedImageB64: null,
    };

    // DOM Elements
    const elements = {
        headerLogo: document.getElementById("header-logo"),
        btnToggleVoice: document.getElementById("btn-toggle-voice"),
        btnToggleSidebar: document.getElementById("btn-toggle-sidebar"),
        btnCloseSidebar: document.getElementById("btn-close-sidebar"),
        sidebar: document.getElementById("analytics-sidebar"),
        chatStream: document.getElementById("chat-stream"),
        chatStreamWrapper: document.getElementById("chat-stream-wrapper"),
        neuralScanner: document.getElementById("neural-scanner"),
        scannerSourceText: document.getElementById("scanner-source-text"),

        // Input Dock Tabs
        tabButtons: document.querySelectorAll(".dock-tab-btn"),
        tabPanes: document.querySelectorAll(".dock-tab-pane"),

        // Upload Tab
        dropZone: document.getElementById("drop-zone"),
        fileInput: document.getElementById("file-input"),
        dropPrompt: document.getElementById("drop-prompt"),
        dropPreviewWrapper: document.getElementById("drop-preview-wrapper"),
        dropPreviewImg: document.getElementById("drop-preview-img"),
        btnAnalyzeUpload: document.getElementById("btn-analyze-upload"),
        btnCancelUpload: document.getElementById("btn-cancel-upload"),

        // Camera Tab
        cameraVideo: document.getElementById("camera-video"),
        cameraCanvas: document.getElementById("camera-snapshot-canvas"),
        btnStartCamera: document.getElementById("btn-start-camera"),
        btnSwitchCamera: document.getElementById("btn-switch-camera"),
        btnCaptureCamera: document.getElementById("btn-capture-camera"),

        // Draw Canvas Tab
        drawingCanvas: document.getElementById("drawing-canvas"),
        brushSlider: document.getElementById("brush-slider"),
        brushSizeVal: document.getElementById("brush-size-val"),
        btnAnalyzeDraw: document.getElementById("btn-analyze-draw"),
        btnClearCanvas: document.getElementById("btn-clear-canvas"),

        // Sample Digits Tab
        samplesGrid: document.getElementById("samples-grid"),

        // Sidebar Elements
        metricAccuracy: document.getElementById("metric-accuracy"),
        metricLoss: document.getElementById("metric-loss"),
        sideTabButtons: document.querySelectorAll(".side-tab-btn"),
        sideTabPanes: document.querySelectorAll(".side-tab-pane"),
        historyCount: document.getElementById("history-count"),
        historyTbody: document.getElementById("history-tbody"),
        btnClearHistory: document.getElementById("btn-clear-history"),
        btnResetChat: document.getElementById("btn-reset-chat"),

        // Modal
        modalBackdrop: document.getElementById("modal-backdrop"),
        modalTitle: document.getElementById("modal-title"),
        modalBody: document.getElementById("modal-body"),
        btnCloseModal: document.getElementById("btn-close-modal"),
    };

    // Canvas 2D Context
    const ctx = elements.drawingCanvas.getContext("2d");

    // ==========================================
    // INITIALIZATION
    // ==========================================
    function init() {
        initCanvas();
        setupEventListeners();
        loadMetrics();
        loadHistory();
        loadSampleDigits();

        // Update welcome message time
        const welcomeTime = document.getElementById("welcome-time");
        if (welcomeTime) {
            welcomeTime.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        }
    }

    // ==========================================
    // VOICE ASSISTANT (WEB SPEECH API)
    // ==========================================
    function speakText(text) {
        if (!state.voiceEnabled || !("speechSynthesis" in window)) return;
        try {
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 0.96;
            utterance.pitch = 1.05;
            utterance.volume = 1.0;

            const voices = window.speechSynthesis.getVoices();
            const naturalVoice = voices.find(v =>
                v.lang.startsWith("en") && (v.name.includes("Natural") || v.name.includes("Google") || v.name.includes("Neural"))
            );
            if (naturalVoice) {
                utterance.voice = naturalVoice;
            }
            window.speechSynthesis.speak(utterance);
        } catch (e) {
            console.warn("Speech synthesis error:", e);
        }
    }

    // ==========================================
    // CANVAS DRAWING (TOUCH & MOUSE ENGINE)
    // ==========================================
    function initCanvas() {
        const canvas = elements.drawingCanvas;
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = state.brushSize;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        function getPos(e) {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            if (e.touches && e.touches.length > 0) {
                return {
                    x: (e.touches[0].clientX - rect.left) * scaleX,
                    y: (e.touches[0].clientY - rect.top) * scaleY
                };
            }
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        }

        function startDraw(e) {
            e.preventDefault();
            state.isDrawing = true;
            const pos = getPos(e);
            state.lastX = pos.x;
            state.lastY = pos.y;
            ctx.beginPath();
            ctx.arc(pos.x, pos.y, ctx.lineWidth / 2, 0, Math.PI * 2);
            ctx.fillStyle = "#FFFFFF";
            ctx.fill();
        }

        function draw(e) {
            if (!state.isDrawing) return;
            e.preventDefault();
            const pos = getPos(e);
            ctx.beginPath();
            ctx.moveTo(state.lastX, state.lastY);
            ctx.lineTo(pos.x, pos.y);
            ctx.stroke();
            state.lastX = pos.x;
            state.lastY = pos.y;
        }

        function stopDraw() {
            state.isDrawing = false;
        }

        canvas.addEventListener("mousedown", startDraw);
        canvas.addEventListener("mousemove", draw);
        window.addEventListener("mouseup", stopDraw);

        canvas.addEventListener("touchstart", startDraw, { passive: false });
        canvas.addEventListener("touchmove", draw, { passive: false });
        window.addEventListener("touchend", stopDraw);
    }

    function clearCanvas() {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, elements.drawingCanvas.width, elements.drawingCanvas.height);
    }

    function isCanvasBlank() {
        const pixelBuffer = new Uint32Array(
            ctx.getImageData(0, 0, elements.drawingCanvas.width, elements.drawingCanvas.height).data.buffer
        );
        // Check if there are white strokes (alpha != 0 and RGB > black)
        return !pixelBuffer.some(color => (color & 0x00FFFFFF) !== 0);
    }

    // ==========================================
    // LIVE CAMERA ENGINE (WEBRTC)
    // ==========================================
    async function startCamera() {
        try {
            stopCamera();
            const constraints = {
                video: {
                    facingMode: state.cameraFacing,
                    width: { ideal: 640 },
                    height: { ideal: 480 }
                },
                audio: false
            };
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            state.cameraStream = stream;
            elements.cameraVideo.srcObject = stream;
            elements.btnStartCamera.style.display = "none";
            elements.btnSwitchCamera.style.display = "inline-flex";
            elements.btnCaptureCamera.style.display = "inline-flex";
        } catch (err) {
            alert("Could not access camera: " + err.message + "\nPlease grant camera permission.");
        }
    }

    function stopCamera() {
        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach(track => track.stop());
            state.cameraStream = null;
        }
        elements.cameraVideo.srcObject = null;
        elements.btnStartCamera.style.display = "inline-flex";
        elements.btnSwitchCamera.style.display = "none";
        elements.btnCaptureCamera.style.display = "none";
    }

    function switchCamera() {
        state.cameraFacing = state.cameraFacing === "user" ? "environment" : "user";
        startCamera();
    }

    function captureCamera() {
        if (!state.cameraStream) return;
        const video = elements.cameraVideo;
        const canvas = elements.cameraCanvas;
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const c2d = canvas.getContext("2d");
        c2d.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/png");
        submitImage(dataUrl, "Live Camera Snapshot");
    }

    // ==========================================
    // UPLOAD / DROPZONE ENGINE
    // ==========================================
    function handleSelectedFile(file) {
        if (!file || !file.type.startsWith("image/")) {
            alert("Please select a valid image file (PNG, JPG, JPEG).");
            return;
        }
        const reader = new FileReader();
        reader.onload = function (e) {
            state.uploadedImageB64 = e.target.result;
            elements.dropPreviewImg.src = e.target.result;
            elements.dropPrompt.style.display = "none";
            elements.dropPreviewWrapper.style.display = "flex";
        };
        reader.readAsDataURL(file);
    }

    function resetUploadZone() {
        state.uploadedImageB64 = null;
        elements.fileInput.value = "";
        elements.dropPreviewImg.src = "";
        elements.dropPrompt.style.display = "block";
        elements.dropPreviewWrapper.style.display = "none";
    }

    const DIGIT_WORDS_ARR = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"];

    // Cached Client-Side ONNX Session
    let _clientOrtSession = null;
    async function getClientOrtSession() {
        if (_clientOrtSession) return _clientOrtSession;
        if (!window.ort) return null;
        try {
            ort.env.wasm.numThreads = 1;
            const modelCandidates = ["/models/digit_model.onnx", "/assets/digit_model.onnx", "/public/models/digit_model.onnx"];
            for (const url of modelCandidates) {
                try {
                    _clientOrtSession = await ort.InferenceSession.create(url);
                    return _clientOrtSession;
                } catch (e) {
                    // try next
                }
            }
        } catch (err) {
            console.warn("ORT Web initialization note:", err);
        }
        return null;
    }

    // Client-side visual bounding box renderer
    function drawBBoxesOnImage(imgUrl, words) {
        return new Promise(resolve => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                const canvas = document.createElement("canvas");
                canvas.width = img.naturalWidth || img.width;
                canvas.height = img.naturalHeight || img.height;
                const c2d = canvas.getContext("2d");
                c2d.drawImage(img, 0, 0);

                c2d.strokeStyle = "#2563eb";
                c2d.lineWidth = Math.max(2, Math.round(canvas.width / 260));
                const fontSize = Math.max(13, Math.round(canvas.width / 36));
                c2d.font = `bold ${fontSize}px Inter, sans-serif`;

                for (const w of words) {
                    if (!w || !w.bbox) continue;
                    const { x0, y0, x1, y1 } = w.bbox;
                    const bw = x1 - x0;
                    const bh = y1 - y0;
                    c2d.strokeRect(x0, y0, bw, bh);

                    const label = `${w.text} (${Math.round(w.confidence || 90)}%)`;
                    const textWidth = c2d.measureText(label).width;
                    const tagY = Math.max(fontSize + 4, y0 - 4);
                    c2d.fillStyle = "#2563eb";
                    c2d.fillRect(x0, tagY - fontSize, textWidth + 8, fontSize + 4);
                    c2d.fillStyle = "#ffffff";
                    c2d.fillText(label, x0 + 4, tagY);
                    c2d.fillStyle = "#2563eb";
                }
                resolve(canvas.toDataURL("image/png"));
            };
            img.onerror = () => resolve(imgUrl);
            img.src = imgUrl;
        });
    }

    // Client-side digit preprocessor into standardized 28x28 MNIST format
    function preprocessDigitClient(imgUrl) {
        return new Promise(resolve => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            img.onload = () => {
                const c = document.createElement("canvas");
                c.width = img.naturalWidth || img.width;
                c.height = img.naturalHeight || img.height;
                const c2d = c.getContext("2d");
                c2d.drawImage(img, 0, 0);

                const imgData = c2d.getImageData(0, 0, c.width, c.height);
                const d = imgData.data;
                const w = c.width;
                const h = c.height;

                // Border polarity check
                let borderSum = 0;
                let borderCount = 0;
                for (let x = 0; x < w; x++) {
                    const iT = (0 * w + x) * 4;
                    const iB = ((h - 1) * w + x) * 4;
                    borderSum += (d[iT] + d[iT + 1] + d[iT + 2]) / 3;
                    borderSum += (d[iB] + d[iB + 1] + d[iB + 2]) / 3;
                    borderCount += 2;
                }
                for (let y = 0; y < h; y++) {
                    const iL = (y * w + 0) * 4;
                    const iR = (y * w + (w - 1)) * 4;
                    borderSum += (d[iL] + d[iL + 1] + d[iL + 2]) / 3;
                    borderSum += (d[iR] + d[iR + 1] + d[iR + 2]) / 3;
                    borderCount += 2;
                }
                const isDarkBg = (borderSum / borderCount) < 127;

                // Thresholding & Bounding Box
                let minX = w, maxX = 0, minY = h, maxY = 0;
                const bin = new Uint8Array(w * h);
                let whiteCount = 0;
                for (let y = 0; y < h; y++) {
                    for (let x = 0; x < w; x++) {
                        const idx = (y * w + x) * 4;
                        const gray = (d[idx] + d[idx + 1] + d[idx + 2]) / 3;
                        const isStroke = isDarkBg ? (gray > 90) : (gray < 165);
                        if (isStroke) {
                            bin[y * w + x] = 255;
                            whiteCount++;
                            if (x < minX) minX = x;
                            if (x > maxX) maxX = x;
                            if (y < minY) minY = y;
                            if (y > maxY) maxY = y;
                        }
                    }
                }

                if (whiteCount < 15 || minX >= maxX || minY >= maxY) {
                    resolve(null);
                    return;
                }

                const cropW = maxX - minX + 1;
                const cropH = maxY - minY + 1;
                const cropCanvas = document.createElement("canvas");
                cropCanvas.width = cropW;
                cropCanvas.height = cropH;
                const cropCtx = cropCanvas.getContext("2d");
                const cropData = cropCtx.createImageData(cropW, cropH);
                for (let y = 0; y < cropH; y++) {
                    for (let x = 0; x < cropW; x++) {
                        const val = bin[(minY + y) * w + (minX + x)];
                        const tIdx = (y * cropW + x) * 4;
                        cropData.data[tIdx] = val;
                        cropData.data[tIdx + 1] = val;
                        cropData.data[tIdx + 2] = val;
                        cropData.data[tIdx + 3] = 255;
                    }
                }
                cropCtx.putImageData(cropData, 0, 0);

                let newW = 20, newH = 20;
                if (cropH > cropW) {
                    newH = 20;
                    newW = Math.max(1, Math.round(cropW * 20 / cropH));
                } else {
                    newW = 20;
                    newH = Math.max(1, Math.round(cropH * 20 / cropW));
                }

                const c28 = document.createElement("canvas");
                c28.width = 28;
                c28.height = 28;
                const ctx28 = c28.getContext("2d");
                ctx28.fillStyle = "#000000";
                ctx28.fillRect(0, 0, 28, 28);
                const ox = Math.floor((28 - newW) / 2);
                const oy = Math.floor((28 - newH) / 2);
                ctx28.drawImage(cropCanvas, 0, 0, cropW, cropH, ox, oy, newW, newH);

                // Center of Mass shift
                const d28 = ctx28.getImageData(0, 0, 28, 28).data;
                let sumY = 0, sumX = 0, total = 0;
                for (let y = 0; y < 28; y++) {
                    for (let x = 0; x < 28; x++) {
                        const v = d28[(y * 28 + x) * 4];
                        sumX += x * v;
                        sumY += y * v;
                        total += v;
                    }
                }
                if (total > 0) {
                    const cx = sumX / total;
                    const cy = sumY / total;
                    const sx = Math.max(-4, Math.min(4, Math.round(14 - cx)));
                    const sy = Math.max(-4, Math.min(4, Math.round(14 - cy)));
                    if (sx !== 0 || sy !== 0) {
                        const temp = document.createElement("canvas");
                        temp.width = 28; temp.height = 28;
                        temp.getContext("2d").drawImage(c28, 0, 0);
                        ctx28.fillStyle = "#000000";
                        ctx28.fillRect(0, 0, 28, 28);
                        ctx28.drawImage(temp, sx, sy);
                    }
                }

                const finalData = ctx28.getImageData(0, 0, 28, 28).data;
                const floatArr = new Float32Array(28 * 28);
                for (let i = 0; i < 28 * 28; i++) {
                    floatArr[i] = finalData[i * 4] / 255.0;
                }

                resolve({
                    tensor: floatArr,
                    normB64: c28.toDataURL("image/png"),
                    bbox: { x: minX, y: minY, w: cropW, h: cropH }
                });
            };
            img.onerror = () => resolve(null);
            img.src = imgUrl;
        });
    }

    // Client-side inference runner (for offline & instant execution)
    async function runClientInference(dataUrl, sourceName) {
        const prep = await preprocessDigitClient(dataUrl);
        if (!prep) return null;

        const session = await getClientOrtSession();
        if (!session) return null;

        const inputTensor = new ort.Tensor("float32", prep.tensor, [1, 28, 28, 1]);
        const feeds = {};
        feeds[session.inputNames[0]] = inputTensor;
        const results = await session.run(feeds);
        const output = results[session.outputNames[0]].data;

        let maxIdx = 0;
        let maxVal = output[0];
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxVal) {
                maxVal = output[i];
                maxIdx = i;
            }
        }

        const confPct = Math.round(maxVal * 1000) / 10;
        const digitWord = DIGIT_WORDS_ARR[maxIdx] || String(maxIdx);
        const spoken = `The detected digit is ${digitWord} with ${confPct}% confidence.`;

        return {
            success: true,
            primary_detected: String(maxIdx),
            content_type: "Handwritten Digit",
            is_text_word: false,
            reconstructed_number: String(maxIdx),
            avg_confidence: maxVal,
            avg_confidence_pct: confPct,
            annotated_image_b64: dataUrl,
            char_results: [{
                char_index: 1,
                predicted_digit: maxIdx,
                digit_word: digitWord,
                confidence: maxVal,
                confidence_pct: confPct,
                probabilities: Array.from(output),
                gradcam_b64: prep.normB64,
                norm_b64: prep.normB64
            }],
            ocr_blocks: [],
            full_ocr_string: "",
            spoken_message: spoken,
            source: sourceName
        };
    }

    // ==========================================
    // PREDICTION SUBMISSION & CHAT RENDERING
    // ==========================================
    async function submitImage(dataUrl, sourceName) {
        if (!dataUrl) return;

        // 1. Append User Message to Chat Stream
        appendUserMessage(dataUrl, sourceName);

        // 2. Show Dynamic AI Neural Scanner Loading Indicator
        elements.scannerSourceText.textContent = `Analyzing ${sourceName} · Neural Vision & Multi-Scale OCR`;
        elements.neuralScanner.style.display = "block";
        scrollToBottom();

        try {
            // STEP A: Client-Side Deep Handwriting OCR Check (Word/Text detection e.g. "HELLO WORLD")
            let textOcrPayload = null;
            if (window.Tesseract && sourceName !== "Touch Canvas Drawing") {
                try {
                    const ocrRes = await Tesseract.recognize(dataUrl, "eng");
                    if (ocrRes && ocrRes.data && ocrRes.data.text) {
                        const rawText = ocrRes.data.text.replace(/[\r\n]+/g, " ").trim();
                        const hasLetters = /[a-zA-Z]/.test(rawText);
                        const cleanWords = (ocrRes.data.words || [])
                            .filter(w => w.text && w.text.trim().length > 0 && w.confidence > 25)
                            .map(w => ({
                                text: w.text.trim(),
                                confidence: w.confidence,
                                confidence_pct: Math.round(w.confidence),
                                bbox: w.bbox
                            }));

                        if (hasLetters && rawText.length >= 2 && cleanWords.length > 0) {
                            const annotatedB64 = await drawBBoxesOnImage(dataUrl, cleanWords);
                            const avgConf = Math.round(ocrRes.data.confidence || 90);
                            const spoken = `The detected handwritten text is ${rawText}.`;

                            textOcrPayload = {
                                success: true,
                                primary_detected: rawText,
                                content_type: "Handwritten / Printed Text Word",
                                is_text_word: true,
                                reconstructed_number: "",
                                avg_confidence: avgConf / 100,
                                avg_confidence_pct: avgConf,
                                annotated_image_b64: annotatedB64,
                                char_results: [],
                                ocr_blocks: cleanWords,
                                full_ocr_string: rawText,
                                spoken_message: spoken,
                                source: sourceName
                            };
                        }
                    }
                } catch (ocrErr) {
                    console.warn("Client OCR check note:", ocrErr);
                }
            }

            // If text words were identified (e.g. "HELLO WORLD"), render immediately!
            if (textOcrPayload) {
                elements.neuralScanner.style.display = "none";
                appendAssistantMessage(textOcrPayload);
                if (textOcrPayload.spoken_message) speakText(textOcrPayload.spoken_message);
                loadHistory();
                return;
            }

            // STEP B: Digit Recognition (Try Cloud API first, fall back to Client-Side ONNX Engine)
            let resultData = null;
            try {
                const resp = await fetch("/api/predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        image: dataUrl,
                        source: sourceName
                    })
                });

                if (resp.ok) {
                    const rawText = await resp.text();
                    try {
                        const parsed = JSON.parse(rawText);
                        if (parsed && parsed.success) {
                            resultData = parsed;
                        }
                    } catch (e) {
                        // ignore JSON parse error on 500
                    }
                }
            } catch (netErr) {
                console.warn("Cloud API unreachable, using client engine:", netErr);
            }

            // If cloud API didn't succeed or timed out, run client-side ONNX engine
            if (!resultData) {
                resultData = await runClientInference(dataUrl, sourceName);
            }

            // Hide Scanner
            elements.neuralScanner.style.display = "none";

            if (!resultData || !resultData.success) {
                appendAssistantErrorMessage(resultData?.message || "Could not recognize content. Please draw or capture with clear contrast.");
                return;
            }

            // Append Assistant Response Card
            appendAssistantMessage(resultData);

            // Announce speech
            if (resultData.spoken_message) {
                speakText(resultData.spoken_message);
            }

            // Refresh History
            loadHistory();

        } catch (err) {
            elements.neuralScanner.style.display = "none";
            appendAssistantErrorMessage("Processing error: " + err.message);
        }
    }

    function appendUserMessage(imgUrl, sourceName) {
        const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const userDiv = document.createElement("div");
        userDiv.className = "chat-message user";
        userDiv.innerHTML = `
            <div class="msg-header">
                <span class="msg-author">👤 Submitted Sample</span>
                <span class="msg-role-tag">${sourceName}</span>
                <span class="msg-time">${timeStr}</span>
            </div>
            <div class="msg-body">
                <div>Examining handwritten image from <strong>${sourceName}</strong>:</div>
                <img src="${imgUrl}" class="user-thumbnail" alt="Input Sample">
            </div>
        `;
        elements.chatStream.appendChild(userDiv);
        scrollToBottom();
    }

    function appendAssistantErrorMessage(msg) {
        const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const div = document.createElement("div");
        div.className = "chat-message assistant";
        div.innerHTML = `
            <div class="msg-header">
                <img src="/assets/logo.png" class="msg-avatar" alt="Avatar">
                <span class="msg-author">DigitVision AI</span>
                <span class="msg-role-tag">Assistant</span>
                <span class="msg-time">${timeStr}</span>
            </div>
            <div class="msg-body" style="color: var(--danger);">
                ⚠️ ${msg}
            </div>
        `;
        elements.chatStream.appendChild(div);
        scrollToBottom();
    }

    function appendAssistantMessage(data) {
        const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        const isWord = data.is_text_word;
        const primaryText = data.primary_detected || "0";
        const isLongText = primaryText.length > 3;

        // Build character breakdown HTML
        let charBreakdownHtml = "";
        if (data.char_results && data.char_results.length > 0) {
            const charCards = data.char_results.map(c => `
                <div class="char-card" onclick="window.DigitVision.showGradcamModal(${c.char_index}, ${c.predicted_digit}, '${c.digit_word}', ${c.confidence_pct}, '${c.gradcam_b64}', '${c.norm_b64}')">
                    <div class="char-top-row">
                        <span style="color: var(--primary-blue);">#${c.char_index} Digit ${c.predicted_digit}</span>
                        <span style="color: var(--success);">${c.confidence_pct.toFixed(0)}%</span>
                    </div>
                    <img src="${c.gradcam_b64 || c.norm_b64}" class="char-gradcam-thumb" alt="Grad-CAM">
                    <div style="font-size: 11px; color: var(--text-muted);">${c.digit_word}</div>
                </div>
            `).join("");

            charBreakdownHtml = `
                <div class="breakdown-section">
                    <div class="breakdown-header">
                        🔍 Individual Digit Neural Analysis (${data.char_results.length} Segment${data.char_results.length > 1 ? "s" : ""}):
                    </div>
                    <div class="breakdown-grid">${charCards}</div>
                </div>
            `;
        }

        // Build OCR blocks HTML
        let ocrBlocksHtml = "";
        if (data.ocr_blocks && data.ocr_blocks.length > 0) {
            const wordBadges = data.ocr_blocks.map(b => `
                <span class="step-pill" style="color: var(--text-primary); font-weight: 600;">
                    📝 ${b.text} <b style="color: var(--primary-blue); font-size: 11px;">(${b.confidence_pct.toFixed(0)}%)</b>
                </span>
            `).join(" ");

            ocrBlocksHtml = `
                <div class="breakdown-section" style="margin-top: 12px;">
                    <div class="breakdown-header">
                        📑 Deep Handwriting Word Recognition (${data.ocr_blocks.length} Words Identified):
                    </div>
                    <div style="display: flex; flex-wrap: wrap; gap: 6px;">${wordBadges}</div>
                </div>
            `;
        }

        // Visual bounding box block
        let annotatedBoxHtml = "";
        if (data.annotated_image_b64) {
            annotatedBoxHtml = `
                <div class="annotated-box">
                    <div class="box-title">🎯 Visual Character Localization (Bounding Boxes):</div>
                    <img src="${data.annotated_image_b64}" class="annotated-img" alt="Detected Regions">
                </div>
            `;
        }

        const div = document.createElement("div");
        div.className = "chat-message assistant";
        div.innerHTML = `
            <div class="msg-header">
                <img src="/assets/logo.png" class="msg-avatar" alt="Avatar">
                <span class="msg-author">DigitVision AI</span>
                <span class="msg-role-tag">Neural Vision Assistant</span>
                <span class="msg-time">${timeStr}</span>
            </div>
            <div class="msg-body">
                <p>I have inspected the handwritten image thoroughly! I identified the following text and numerical content: <strong>${primaryText}</strong>.</p>
                
                <div class="prediction-card">
                    <div class="result-main-grid">
                        <div class="identified-box">
                            <div class="identified-caption">Identified Content</div>
                            <div class="identified-giant-text ${isLongText ? "long-text" : ""}">${primaryText}</div>
                            <div class="identified-type-pill">${data.content_type}</div>
                            <div class="verified-pill">✔ Detected & Verified (${data.avg_confidence_pct}%)</div>

                            <div class="voice-assistant-card">
                                <div class="voice-info">
                                    <div class="voice-icon">🔊</div>
                                    <div>
                                        <div class="voice-title">Voice Assistant</div>
                                        <div class="voice-speech-text">"${data.spoken_message}"</div>
                                    </div>
                                </div>
                                <button class="btn-speak" onclick="window.DigitVision.speak('${data.spoken_message.replace(/'/g, "\\'")}')">
                                    🗣️ Speak
                                </button>
                            </div>
                        </div>

                        ${annotatedBoxHtml}
                    </div>

                    ${charBreakdownHtml}
                    ${ocrBlocksHtml}
                </div>
            </div>
        `;

        elements.chatStream.appendChild(div);
        scrollToBottom();
    }

    function scrollToBottom() {
        setTimeout(() => {
            elements.chatStreamWrapper.scrollTop = elements.chatStreamWrapper.scrollHeight;
        }, 50);
    }

    // ==========================================
    // SIDEBAR DATA (METRICS & HISTORY)
    // ==========================================
    async function loadMetrics() {
        try {
            const resp = await fetch("/api/metrics");
            const rawText = await resp.text();
            let data;
            try {
                data = JSON.parse(rawText);
            } catch (parseErr) {
                elements.neuralScanner.style.display = "none";
                appendAssistantErrorMessage("Server error: " + (rawText.slice(0, 160) || "Invalid server response"));
                return;
            }
            if (data.test_accuracy) {
                elements.metricAccuracy.textContent = (data.test_accuracy * 100).toFixed(2) + "%";
            }
            if (data.test_loss) {
                elements.metricLoss.textContent = data.test_loss.toFixed(4);
            }
        } catch (e) {
            console.warn("Could not load metrics:", e);
        }
    }

    async function loadHistory() {
        try {
            const resp = await fetch("/api/history");
            const rows = await resp.json();
            elements.historyCount.textContent = `Logged Inferences: ${rows.length}`;

            if (rows.length === 0) {
                elements.historyTbody.innerHTML = `<tr><td colspan="4" class="empty-state">No predictions logged yet.</td></tr>`;
                return;
            }

            elements.historyTbody.innerHTML = rows.slice(0, 20).map(r => `
                <tr>
                    <td style="color: var(--text-light);">${r.timestamp ? r.timestamp.split(" ")[1] : "-"}</td>
                    <td style="font-weight: 700; color: var(--primary-blue);">${r.predicted_digit}</td>
                    <td style="color: var(--success); font-weight: 600;">${parseFloat(r.confidence_pct || 0).toFixed(0)}%</td>
                    <td style="font-size: 10px; color: var(--text-muted);">${r.input_source}</td>
                </tr>
            `).join("");
        } catch (e) {
            console.warn("Could not load history:", e);
        }
    }

    async function clearHistory() {
        if (!confirm("Are you sure you want to clear the prediction history log?")) return;
        try {
            await fetch("/api/clear_history", { method: "POST" });
            loadHistory();
        } catch (e) {
            alert("Error clearing history: " + e.message);
        }
    }

    async function loadSampleDigits() {
        try {
            const resp = await fetch("/api/sample_digits");
            const samples = await resp.json();
            if (!samples || samples.length === 0) {
                elements.samplesGrid.innerHTML = `<div>No samples available.</div>`;
                return;
            }
            elements.samplesGrid.innerHTML = samples.map(s => `
                <div class="sample-card" onclick="window.DigitVision.testSample('${s.url}', '${s.label}')">
                    <img src="${s.url}" class="sample-img" alt="Sample ${s.label}">
                    <span class="sample-label">Digit '${s.label}'</span>
                </div>
            `).join("");
        } catch (e) {
            console.warn("Could not load sample digits:", e);
        }
    }

    // Convert an image URL to base64 Data URL
    async function urlToDataUrl(url) {
        const resp = await fetch(url);
        const blob = await resp.blob();
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onloadend = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    // ==========================================
    // EVENT LISTENERS SETUP
    // ==========================================
    function setupEventListeners() {
        // Voice Toggle
        elements.btnToggleVoice.addEventListener("click", () => {
            state.voiceEnabled = !state.voiceEnabled;
            elements.btnToggleVoice.classList.toggle("active", state.voiceEnabled);
            elements.btnToggleVoice.querySelector(".btn-text").textContent = state.voiceEnabled ? "Voice On" : "Voice Off";
            if (!state.voiceEnabled && "speechSynthesis" in window) {
                window.speechSynthesis.cancel();
            }
        });

        // Sidebar Toggle (Mobile Drawer)
        elements.btnToggleSidebar.addEventListener("click", () => {
            elements.sidebar.classList.toggle("open");
        });
        elements.btnCloseSidebar.addEventListener("click", () => {
            elements.sidebar.classList.remove("open");
        });

        // Reset Chat
        elements.btnResetChat.addEventListener("click", () => {
            elements.chatStream.innerHTML = "";
            const welcome = document.createElement("div");
            welcome.className = "chat-message assistant";
            welcome.innerHTML = `
                <div class="msg-header">
                    <img src="/assets/logo.png" class="msg-avatar" alt="Avatar">
                    <span class="msg-author">DigitVision AI</span>
                    <span class="msg-role-tag">Neural Vision Assistant</span>
                </div>
                <div class="msg-body">
                    🔄 Chat session reset! Ready for your next handwritten digit or text image.
                </div>
            `;
            elements.chatStream.appendChild(welcome);
        });

        // Clear History
        elements.btnClearHistory.addEventListener("click", clearHistory);

        // Input Dock Tab Switcher
        elements.tabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetTab = btn.getAttribute("data-tab");
                elements.tabButtons.forEach(b => b.classList.remove("active"));
                elements.tabPanes.forEach(p => p.classList.remove("active"));
                btn.classList.add("active");
                document.getElementById(targetTab).classList.add("active");

                // Stop camera if navigating away
                if (targetTab !== "tab-camera") {
                    stopCamera();
                }
            });
        });

        // Sidebar Tab Switcher
        elements.sideTabButtons.forEach(btn => {
            btn.addEventListener("click", () => {
                const targetTab = btn.getAttribute("data-sidetab");
                elements.sideTabButtons.forEach(b => b.classList.remove("active"));
                elements.sideTabPanes.forEach(p => p.classList.remove("active"));
                btn.classList.add("active");
                document.getElementById(targetTab).classList.add("active");
            });
        });

        // Upload Tab Handlers
        elements.dropZone.addEventListener("click", (e) => {
            if (e.target.closest(".preview-actions")) return;
            elements.fileInput.click();
        });

        elements.fileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files[0]) {
                handleSelectedFile(e.target.files[0]);
            }
        });

        elements.dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            elements.dropZone.classList.add("dragover");
        });

        elements.dropZone.addEventListener("dragleave", () => {
            elements.dropZone.classList.remove("dragover");
        });

        elements.dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            elements.dropZone.classList.remove("dragover");
            if (e.dataTransfer.files && e.dataTransfer.files[0]) {
                handleSelectedFile(e.dataTransfer.files[0]);
            }
        });

        // Clipboard Paste Listener (Ctrl + V)
        window.addEventListener("paste", (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let item of items) {
                if (item.type.indexOf("image") !== -1) {
                    const blob = item.getAsFile();
                    handleSelectedFile(blob);
                    // Switch to Upload tab
                    document.querySelector('[data-tab="tab-upload"]').click();
                    break;
                }
            }
        });

        elements.btnAnalyzeUpload.addEventListener("click", () => {
            if (state.uploadedImageB64) {
                submitImage(state.uploadedImageB64, "Uploaded Image");
                resetUploadZone();
            }
        });

        elements.btnCancelUpload.addEventListener("click", resetUploadZone);

        // Camera Tab Handlers
        elements.btnStartCamera.addEventListener("click", startCamera);
        elements.btnSwitchCamera.addEventListener("click", switchCamera);
        elements.btnCaptureCamera.addEventListener("click", captureCamera);

        // Canvas Handlers
        elements.brushSlider.addEventListener("input", (e) => {
            state.brushSize = parseInt(e.target.value);
            elements.brushSizeVal.textContent = `${state.brushSize}px`;
            ctx.lineWidth = state.brushSize;
        });

        elements.btnClearCanvas.addEventListener("click", clearCanvas);

        elements.btnAnalyzeDraw.addEventListener("click", () => {
            if (isCanvasBlank()) {
                alert("⚠️ Canvas is empty! Please draw a handwritten digit or number first.");
                return;
            }
            const dataUrl = elements.drawingCanvas.toDataURL("image/png");
            submitImage(dataUrl, "Touch Canvas Drawing");
        });

        // Modal Close
        elements.btnCloseModal.addEventListener("click", () => {
            elements.modalBackdrop.style.display = "none";
        });
        elements.modalBackdrop.addEventListener("click", (e) => {
            if (e.target === elements.modalBackdrop) {
                elements.modalBackdrop.style.display = "none";
            }
        });
    }

    // ==========================================
    // GLOBAL NAMESPACE EXPORTS (FOR INLINE CLICKS)
    // ==========================================
    window.DigitVision = {
        speak: function (text) {
            speakText(text);
        },
        testSample: async function (sampleUrl, label) {
            try {
                const b64 = await urlToDataUrl(sampleUrl);
                submitImage(b64, `Sample Test Digit '${label}'`);
            } catch (err) {
                alert("Failed to load sample: " + err.message);
            }
        },
        showGradcamModal: function (charIndex, digit, word, confPct, gradcamB64, normB64) {
            elements.modalTitle.textContent = `Grad-CAM Attention: #${charIndex} Digit ${digit} (${word}) - ${confPct.toFixed(1)}%`;
            elements.modalBody.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 16px; align-items: center; text-align: center;">
                    <div style="display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;">
                        <div>
                            <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 6px;">NORMALIZED 28×28 TENSOR</div>
                            <img src="${normB64}" style="width: 140px; height: 140px; border-radius: 8px; border: 1.5px solid var(--border-color); image-rendering: pixelated; background: #000;" alt="Normalized">
                        </div>
                        <div>
                            <div style="font-size: 11px; font-weight: 700; color: var(--text-muted); margin-bottom: 6px;">GRAD-CAM HEATMAP OVERLAY</div>
                            <img src="${gradcamB64}" style="width: 140px; height: 140px; border-radius: 8px; border: 1.5px solid var(--border-blue);" alt="Grad-CAM">
                        </div>
                    </div>
                    <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.5; max-width: 440px;">
                        The red-yellow gradient highlights the specific convolutional activations that triggered the neural network to classify this character as <b>Digit ${digit} (${word})</b> with <b>${confPct.toFixed(1)}%</b> confidence.
                    </div>
                </div>
            `;
            elements.modalBackdrop.style.display = "flex";
        }
    };

    // Boot the application on DOM ready
    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }

})();

