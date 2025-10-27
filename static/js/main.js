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
     * Inizializza l'accesso alla camera e avvia lo streaming.
     * @returns {Promise<boolean>}
     */
    async initializeCameraWithFallback(preferredDeviceId = null) {
    console.log('Tentativo di accesso alla camera con fallback...');

        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showStatus('API camera non supportata dal browser', 'error');
            return false;
        }

        // Helper: detect Safari (incluso iOS)
        const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
        const baseConstraints = isSafari ? { video: true } : {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'user'
            }
        };

        // If a specific deviceId is given, prefer that
        const tryConstraintsList = [];

        if (preferredDeviceId) {
            tryConstraintsList.push({ video: { deviceId: { exact: preferredDeviceId } } });
        }

        // normal constraints (may be rejected by Safari)
        tryConstraintsList.push(baseConstraints);

        // minimal constraint as last resort
        tryConstraintsList.push({ video: true });

        // enumerate devices to allow UI selection if needed
        let devices = [];
        try {
            devices = await navigator.mediaDevices.enumerateDevices();
            const cams = devices.filter(d => d.kind === 'videoinput');
            if (cams.length > 1 && !preferredDeviceId) {
                console.log('Multiple cameras detected, using default or first available');
                // optional: present UI to choose cam, here we pick first
                tryConstraintsList.unshift({ video: { deviceId: { exact: cams[0].deviceId } } });
            }
        } catch (e) {
            console.warn('Impossibile enumerare dispositivi:', e);
        }

        // Try constraints sequentially
        for (const constraints of tryConstraintsList) {
            try {
                console.log('Provando constraint:', constraints);
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                // stop previous stream if exists
                if (window.sensorApp && window.sensorApp.stream) {
                    window.sensorApp.destroy();
                }
                // attach stream
                if (!document.getElementById('videoElement')) {
                    console.error('Elemento video non trovato');
                    stream.getTracks().forEach(t => t.stop());
                    return false;
                }

                document.getElementById('videoElement').srcObject = stream;
                // save stream
                if (window.sensorApp) {
                    window.sensorApp.stream = stream;
                } else {
                    // fallback: set global stream
                    window._tempStream = stream;
                }

                // wait metadata
                await new Promise(resolve => {
                    const v = document.getElementById('videoElement');
                    const loaded = () => { v.removeEventListener('loadedmetadata', loaded); resolve(); };
                    v.addEventListener('loadedmetadata', loaded);
                    setTimeout(resolve, 2500);
                });

                console.log('Camera inizializzata con successo');
                return true;
            } catch (err) {
                console.warn('getUserMedia fallito con constraint:', constraints, err);
                if (err && err.name === 'OverconstrainedError') {
                    console.warn('OverconstrainedError: rimuovo vincoli e riprovo');
                    continue;
                }
                if (err && err.name === 'NotAllowedError') {
                    showStatus('Permesso camera negato. Controlla le impostazioni del browser', 'error');
                    return false;
                }
                // altrimenti prova prossima opzione
            }
    }

    showStatus('Errore: impossibile inizializzare la fotocamera', 'error');
    return false;
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
            app.initializeCameraWithFallback().then(success => {
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