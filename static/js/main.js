"use strict";

class SensorValidationApp {
    constructor() {
        this.isValidating = false;
        this.sessionId = 'default';
        this.stream = null;
        this.referenceLoaded = false;

        this.initializeElements();
        this.setupEventListeners();
        this.generateSessionId();
        this.checkReferenceStatus();
    }

    initializeElements() {
        this.videoElement = document.getElementById('videoElement');
        this.canvasElement = document.getElementById('overlayCanvas');
        this.context = this.canvasElement ? this.canvasElement.getContext('2d') : null;

        this.startButton = document.getElementById('startBtn');
        this.stopButton = document.getElementById('stopBtn');
        this.statusElement = document.getElementById('statusMessage');
        this.referencePreview = document.getElementById('referencePreview');

        // --- CONFIGURAZIONE MOBILE ---
        if (this.videoElement) {
            // 'playsinline' obbligatorio per iOS per non andare in fullscreen
            this.videoElement.setAttribute('playsinline', '');
            this.videoElement.setAttribute('webkit-playsinline', '');
            // 'muted' spesso necessario per autoplay su mobile
            this.videoElement.muted = true;
        }

        this.verifyDOMElements();
    }

    verifyDOMElements() {
        const requiredElements = {
            videoElement: this.videoElement,
            canvasElement: this.canvasElement,
            startButton: this.startButton,
            stopButton: this.stopButton,
            statusElement: this.statusElement
        };

        for (const [name, element] of Object.entries(requiredElements)) {
            if (!element) {
                console.warn(`Elemento DOM non trovato: ${name}`);
            }
        }
    }

    setupEventListeners() {
        // Event listeners gestiti nell'HTML
    }

    async checkReferenceStatus() {
        try {
            const localOverlay = localStorage.getItem('validationOverlayImage');
            const localMetadata = localStorage.getItem('referenceMetadata');

            if (localOverlay && localMetadata) {
                console.log('✅ Riferimento trovato nel localStorage');
                this.referenceLoaded = true;
            } else {
                console.log('❌ Nessun riferimento trovato');
                this.referenceLoaded = false;
            }
        } catch (error) {
            console.warn('Errore check riferimento:', error);
            this.referenceLoaded = false;
        }
    }

    generateSessionId() {
        const urlParams = new URLSearchParams(window.location.search);
        const urlSessionId = urlParams.get('session_id');

        if (urlSessionId) {
            this.sessionId = urlSessionId;
        } else {
            const storedSessionId = localStorage.getItem('sensor_validation_session');
            if (storedSessionId) {
                this.sessionId = storedSessionId;
            } else {
                this.sessionId = 'session_' + Math.random().toString(36).substring(2, 9);
                localStorage.setItem('sensor_validation_session', this.sessionId);
            }
        }
    }

    calculateOptimalResolution(containerWidth, containerHeight, referenceAspect = null) {
        const targetAspect = referenceAspect || (containerWidth / containerHeight);

        const MAX_WIDTH = 1920;
        const MAX_HEIGHT = 1080;

        let width, height;

        if (targetAspect > 1) { // Landscape
            width = Math.min(containerWidth * window.devicePixelRatio, MAX_WIDTH);
            height = Math.round(width / targetAspect);
        } else { // Portrait
            height = Math.min(containerHeight * window.devicePixelRatio, MAX_HEIGHT);
            width = Math.round(height * targetAspect);
        }

        // Dimensioni pari
        width = Math.floor(width / 2) * 2;
        height = Math.floor(height / 2) * 2;

        return {width, height};
    }

    /**
     * Inizializza camera con PRIORITÀ FOTOCAMERA POSTERIORE
     */
    async initializeCameraWithAdaptiveResolution(preferredDeviceId = null, referenceImage = null) {
        console.log('Inizializzazione camera...');

        // 1. Check Sicurezza (HTTPS)
        const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
        if (!window.isSecureContext && !isLocalhost) {
            const msg = 'ERRORE SICUREZZA: La camera richiede HTTPS. Usa "flask run --cert=adhoc" o ngrok.';
            console.error(msg);
            this.showStatus('Errore: Serve HTTPS', 'error');
            alert(msg);
            return false;
        }

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showStatus('API camera non supportata', 'error');
            return false;
        }

        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);

        const videoContainer = document.querySelector('.video-container');
        let containerWidth = videoContainer ? videoContainer.clientWidth : window.innerWidth;
        let containerHeight = videoContainer ? videoContainer.clientHeight : window.innerHeight;

        let referenceAspect = null;
        if (referenceImage && referenceImage.naturalWidth && referenceImage.naturalHeight) {
            referenceAspect = referenceImage.naturalWidth / referenceImage.naturalHeight;
        }

        const optimalRes = this.calculateOptimalResolution(containerWidth, containerHeight, referenceAspect);

        // LISTA TENTATIVI (Priorità alla Posteriore)
        const tryConstraintsList = [];

        // Tentativo 1: Device ID specifico (se selezionato manualmente)
        if (preferredDeviceId) {
            tryConstraintsList.push({
                video: {
                    deviceId: {exact: preferredDeviceId},
                    width: {ideal: optimalRes.width},
                    height: {ideal: optimalRes.height}
                }
            });
        }

        // Tentativo 2: Alta risoluzione + Preferenza Posteriore
        if (!isMobile) {
            tryConstraintsList.push({
                video: {
                    width: {ideal: optimalRes.width},
                    height: {ideal: optimalRes.height},
                    // { ideal: 'environment' } prova la posteriore, ma se sei su PC usa la webcam normale
                    facingMode: {ideal: 'environment'}
                }
            });
        }

        // Tentativo 3: Mobile Friendly + OBBLIGO POSTERIORE
        tryConstraintsList.push({
            video: {
                // MODIFICA CRUCIALE: 'environment' forza la camera posteriore
                facingMode: 'environment',
                width: {ideal: isMobile ? 1280 : optimalRes.width},
                height: {ideal: isMobile ? 720 : optimalRes.height}
            }
        });

        // Tentativo 4: Fallback generico (se la posteriore fallisce, prende qualsiasi cosa)
        tryConstraintsList.push({video: true});

        // Loop esecuzione tentativi
        for (const constraints of tryConstraintsList) {
            try {
                console.log('Provo constraint:', JSON.stringify(constraints));
                const stream = await navigator.mediaDevices.getUserMedia(constraints);

                if (this.stream) this.stream.getTracks().forEach(t => t.stop());

                if (!this.videoElement) throw new Error('Video element missing');

                this.videoElement.srcObject = stream;
                this.stream = stream;

                // Promise play() esplicita
                try {
                    await this.videoElement.play();
                } catch (playError) {
                    console.warn("Autoplay bloccato, riprovo:", playError);
                }

                if (this.videoElement.readyState < 1) {
                    await new Promise(r => this.videoElement.onloadedmetadata = r);
                }

                const videoTrack = stream.getVideoTracks()[0];
                const settings = videoTrack.getSettings();
                console.log(`✅ Camera attiva: ${settings.width}x${settings.height}`);

                const camStatus = document.getElementById('camStatus');
                if (camStatus) {
                    camStatus.textContent = `Attiva (${settings.width}x${settings.height})`;
                    camStatus.className = 'badge bg-success';
                }

                this.showStatus('Camera pronta', 'success');
                return true;

            } catch (err) {
                console.warn('Tentativo fallito:', err.name);
                if (err.name === 'NotAllowedError') {
                    this.showStatus('Permesso camera negato', 'error');
                    return false;
                }
            }
        }

        this.showStatus('Impossibile avviare la fotocamera', 'error');
        return false;
    }

    async updateCameraResolution(referenceImage = null) {
        if (!this.stream) return false;
        const videoTrack = this.stream.getVideoTracks()[0];
        const deviceId = videoTrack.getSettings().deviceId;
        return await this.initializeCameraWithAdaptiveResolution(deviceId, referenceImage);
    }

    clearCanvas() {
        if (this.context && this.canvasElement) {
            this.context.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        }
    }

    showStatus(message, type) {
        if (this.statusElement) {
            this.statusElement.textContent = message;
            this.statusElement.className = `ms-3 text-${type}`;
        }
    }

    destroy() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    }
}

// --- INITIALIZATION ---
try {
    console.log('🚀 Init SensorValidationApp...');
    const app = new SensorValidationApp();
    window.sensorApp = app;

    setTimeout(() => {
        const storedReference = localStorage.getItem('validationOverlayImage');
        let referenceImage = null;

        if (storedReference) {
            referenceImage = new Image();
            referenceImage.src = storedReference;
        }

        const initializeCam = (img) => {
            app.initializeCameraWithAdaptiveResolution(null, img).then(success => {
                if (!success) app.showStatus('Premi Start per avviare', 'warning');
            });
        };

        if (referenceImage) {
            referenceImage.onload = () => initializeCam(referenceImage);
            referenceImage.onerror = () => initializeCam(null);
        } else {
            initializeCam(null);
        }

    }, 500);

} catch (error) {
    console.error('❌ Init failed:', error);
}