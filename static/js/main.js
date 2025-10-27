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
            } else {
                console.log(`✓ Elemento trovato: ${name}`);
            }
        }
    }

    setupEventListeners() {
        // Event listeners vengono gestiti direttamente nella pagina HTML
        // per evitare conflitti con il sistema di overlay
    }

    async checkReferenceStatus() {
        try {
            console.log('Checking reference status...');
            const localReference = localStorage.getItem('referenceImage');
            const localMetadata = localStorage.getItem('referenceMetadata');

            if (localReference && localMetadata) {
                const metadata = JSON.parse(localMetadata);
                console.log('✅ Riferimento trovato nel localStorage');
                this.referenceLoaded = true;
                this.showStatus('Riferimento caricato con successo! Pronto per la validazione.', 'success');

                if (this.referencePreview) {
                    this.displayReferencePreview(localReference, metadata);
                }

                if (this.startButton) {
                    this.startButton.disabled = false;
                }
                return true;
            }

            console.log('❌ Nessun riferimento trovato nel localStorage');
            this.referenceLoaded = false;
            this.showReferenceRequiredMessage();
            return false;

        } catch (error) {
            console.warn('Errore nel verificare lo stato del riferimento:', error);
            this.referenceLoaded = false;
            this.showReferenceRequiredMessage();
            return false;
        }
    }

    showReferenceRequiredMessage() {
        this.showStatus('Carica un\'immagine di riferimento per iniziare', 'warning');

        if (this.referencePreview) {
            this.referencePreview.innerHTML = `
                <div class="card-body text-center">
                    <i class="bi bi-exclamation-triangle fs-1 text-warning mb-3"></i>
                    <h5>Nessun Riferimento Caricato</h5>
                    <p class="text-muted">Per avviare la validazione, devi prima caricare un'immagine di riferimento.</p>
                    <a href="upload.html" class="btn btn-primary mt-2">
                        <i class="bi bi-upload"></i> Carica Riferimento
                    </a>
                </div>
            `;
        }
    }

    displayReferencePreview(imageData, metadata) {
        if (!this.referencePreview) return;

        const imgSrc = imageData.startsWith('data:') ? imageData : `data:image/jpeg;base64,${imageData}`;

        this.referencePreview.innerHTML = `
            <div class="card-body">
                <img src="${imgSrc}" alt="Reference preview" class="img-fluid rounded mb-2" 
                     style="max-height: 200px;">
                <p class="mb-1"><strong>Tipo:</strong> ${metadata.limbType === 'arm' ? 'Braccio' : 'Gamba'}</p>
                ${metadata.points_detected ? `<p class="mb-1"><small class="text-muted">Punti anatomici: ${metadata.points_detected}</small></p>` : ''}
                ${metadata.uploadDate ? `<p class="mb-0"><small class="text-muted">Caricato: ${new Date(metadata.uploadDate).toLocaleString()}</small></p>` : ''}
            </div>
        `;
    }

    generateSessionId() {
        const urlParams = new URLSearchParams(window.location.search);
        const urlSessionId = urlParams.get('session_id');

        if (urlSessionId) {
            this.sessionId = urlSessionId;
            console.log('Session ID dall\'URL:', this.sessionId);
        } else {
            const storedSessionId = localStorage.getItem('sensor_validation_session');
            if (storedSessionId) {
                this.sessionId = storedSessionId;
                console.log('Session ID dal localStorage:', this.sessionId);
            } else {
                this.sessionId = 'session' + Math.random().toString(36).substring(2, 9);
                localStorage.setItem('sensor_validation_session', this.sessionId);
                console.log('Nuovo Session ID generato:', this.sessionId);
            }
        }
    }

    /**
     * Calcola la risoluzione ottimale basata sul container e sull'immagine di riferimento
     */
    calculateOptimalResulution(containerWidth, containerHeight, referenceAspect = null) {
        const targetAspect = referenceAspect || (containerWidth / containerHeight);

        const MAX_WIDTH = 1920;
        const MAX_HEIGHT = 1080;

        let width, height;

        if (targetAspect > 1) {
            width = Math.min(containerWidth * window.devicePixelRatio, MAX_WIDTH);
            height = Math.round(width / targetAspect);
        } else {
            // Portrait
            height = Math.min(containerHeight * window.devicePixelRatio, MAX_HEIGHT);
            width = Math.round(height * targetAspect);
        }

        width = Math.floor(width / 2) * 2;
        height = Math.floor(height / 2) * 2;

        return { width, height };
    }


    /**
     * Inizializza l'accesso alla camera e avvia lo streaming.
     * @returns {Promise<boolean>}
     */
    async initializeCameraWithAdaptiveResolution(preferredDeviceId = null, referenceImage = null) {
        console.log('Inizializzazione camera con risoluzione adattiva...');

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            this.showStatus('API camera non supportata dal browser', 'error');
            return false;
        }

        // Ottieni dimensioni del container
        const videoContainer = document.querySelector('.video-container');
        if (!videoContainer) {
            console.error('Container video non trovato');
            return false;
        }

        const containerWidth = videoContainer.clientWidth;
        const containerHeight = videoContainer.clientHeight;

        // Calcola aspect ratio di riferimento se disponibile
        let referenceAspect = null;
        if (referenceImage && referenceImage.naturalWidth && referenceImage.naturalHeight) {
            referenceAspect = referenceImage.naturalWidth / referenceImage.naturalHeight;
        }

        // Calcola risoluzione ottimale
        const optimalRes = this.calculateOptimalResolution(
            containerWidth,
            containerHeight,
            referenceAspect
        );

        console.log(`Risoluzione target: ${optimalRes.width}x${optimalRes.height}`);

        // Detect Safari
        const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);

        // Costruisci lista di constraint da provare
        const tryConstraintsList = [];

        // 1. Constraint con risoluzione specifica e deviceId (se fornito)
        if (preferredDeviceId) {
            tryConstraintsList.push({
                video: {
                    deviceId: { exact: preferredDeviceId },
                    width: { ideal: optimalRes.width },
                    height: { ideal: optimalRes.height },
                    aspectRatio: { ideal: referenceAspect || (optimalRes.width / optimalRes.height) }
                }
            });
        }

        // 2. Constraint con risoluzione specifica (senza deviceId)
        if (!isSafari) {
            tryConstraintsList.push({
                video: {
                    width: { ideal: optimalRes.width },
                    height: { ideal: optimalRes.height },
                    aspectRatio: { ideal: referenceAspect || (optimalRes.width / optimalRes.height) },
                    facingMode: 'user'
                }
            });

            // 3. Constraint con risoluzione min/max
            tryConstraintsList.push({
                video: {
                    width: { min: 640, ideal: optimalRes.width, max: 1920 },
                    height: { min: 480, ideal: optimalRes.height, max: 1080 },
                    facingMode: 'user'
                }
            });
        }

        // 4. Fallback base per Safari
        tryConstraintsList.push({ video: true });

        // Enumera dispositivi disponibili
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cameras = devices.filter(d => d.kind === 'videoinput');

            if (cameras.length > 1 && !preferredDeviceId) {
                console.log(`${cameras.length} camere trovate`);
                // Prova con la prima camera disponibile
                tryConstraintsList.unshift({
                    video: {
                        deviceId: { exact: cameras[0].deviceId },
                        width: { ideal: optimalRes.width },
                        height: { ideal: optimalRes.height }
                    }
                });
            }
        } catch (e) {
            console.warn('Impossibile enumerare dispositivi:', e);
        }

        // Prova i constraint in sequenza
        for (const constraints of tryConstraintsList) {
            try {
                console.log('Tentativo con constraint:', constraints);
                const stream = await navigator.mediaDevices.getUserMedia(constraints);

                // Ferma stream precedente se esiste
                if (this.stream) {
                    this.stream.getTracks().forEach(t => t.stop());
                }

                // Ottieni le impostazioni effettive dello stream
                const videoTrack = stream.getVideoTracks()[0];
                const settings = videoTrack.getSettings();
                console.log('Risoluzione ottenuta:', settings.width, 'x', settings.height);

                // Attacca lo stream al video element
                const videoElement = document.getElementById('videoElement');
                if (!videoElement) {
                    console.error('Elemento video non trovato');
                    stream.getTracks().forEach(t => t.stop());
                    return false;
                }

                videoElement.srcObject = stream;
                this.stream = stream;

                // Aspetta il caricamento dei metadata
                await new Promise((resolve) => {
                    const onLoaded = () => {
                        videoElement.removeEventListener('loadedmetadata', onLoaded);
                        resolve();
                    };
                    videoElement.addEventListener('loadedmetadata', onLoaded);
                    setTimeout(resolve, 2500); // Timeout di sicurezza
                });

                console.log('✅ Camera inizializzata con risoluzione adattiva');

                // Aggiorna badge
                const camStatus = document.getElementById('camStatus');
                if (camStatus) {
                    camStatus.textContent = `Attiva (${settings.width}x${settings.height})`;
                    camStatus.className = 'badge bg-success';
                }

                return true;

            } catch (err) {
                console.warn('Tentativo fallito:', err.name, err.message);

                if (err.name === 'NotAllowedError') {
                    this.showStatus('Permesso camera negato', 'error');
                    return false;
                }
                // Continua con il prossimo constraint
            }
        }

        this.showStatus('Impossibile inizializzare la fotocamera', 'error');
        return false;
    }

    /**
     * Aggiorna la risoluzione della camera quando il container cambia dimensione
     */
    async updateCameraResolution(referenceImage = null) {
        if (!this.stream) {
            console.log('Nessuno stream attivo da aggiornare');
            return false;
        }

        console.log('Aggiornamento risoluzione camera...');

        // Ottieni il deviceId corrente
        const videoTrack = this.stream.getVideoTracks()[0];
        const currentSettings = videoTrack.getSettings();
        const deviceId = currentSettings.deviceId;

        // Re-inizializza con la nuova risoluzione
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
        console.log(`Status [${type}]: ${message}`);
    }

    destroy() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
    }
}

// Inizializzazione globale
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('🚀 Inizializzazione SensorValidationApp...');
        const app = new SensorValidationApp();
        window.sensorApp = app;

        // Inizializzazione camera automatica dopo un breve delay
        setTimeout(() => {
            console.log('Tentativo di inizializzazione camera...');
            app.initializeCameraWithAdaptiveResolution().then(success => {
                if (success) {
                    console.log('✅ Camera initialized successfully');

                    // Aggiorna badge stato camera
                    const camStatus = document.getElementById('camStatus');
                    if (camStatus) {
                        camStatus.textContent = 'Attiva';
                        camStatus.className = 'badge bg-success';
                    }

                    app.showStatus('Camera pronta per la validazione', 'success');
                } else {
                    console.error('❌ Failed to initialize camera');
                    app.showStatus('Errore camera - verifica i permessi e ricarica la pagina', 'error');
                }
            }).catch(error => {
                console.error('❌ Camera initialization error:', error);
                app.showStatus('Errore inizializzazione camera', 'error');
            });
        }, 500);

    } catch (error) {
        console.error('❌ Failed to initialize app:', error);
        const statusElement = document.getElementById('statusMessage');
        if (statusElement) {
            statusElement.textContent = 'Errore di inizializzazione dell\'applicazione';
            statusElement.className = 'ms-3 text-error';
        }
    }
});