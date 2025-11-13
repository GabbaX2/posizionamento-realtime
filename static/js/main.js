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
                // Rimosso log per pulizia console
                // console.log(`✓ Elemento trovato: ${name}`);
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

            // --- MODIFICA CORRETTIVA ---
            // Le chiavi corrette che validation.html si aspetta sono:
            // 'validationOverlayImage' (per il canvas)
            // 'cleanReferenceImage' (per il backend)
            // 'referenceMetadata'
            const localOverlay = localStorage.getItem('validationOverlayImage');
            const localMetadata = localStorage.getItem('referenceMetadata');

            // C'è un'incoerenza. Inizializziamo this.referenceLoaded
            // qui, ma validation.html ha la sua logica per
            // caricare i dati. Ci affidiamo alla logica di validation.html
            // e qui facciamo solo un controllo superficiale.

            if (localOverlay && localMetadata) {
                console.log('✅ Riferimento (overlay/metadata) trovato nel localStorage');
                this.referenceLoaded = true;

                // NOTA: 'validation.html' ha la sua logica 'loadReferenceFromStorage'
                // che è più completa e gestisce l'UI.
                // Questa funzione in main.js è quasi ridondante
                // se non per il this.referenceLoaded.
                // Evitiamo di duplicare la logica di 'displayReferencePreview'
                // che è già gestita meglio in validation.html

            } else {
                console.log('❌ Nessun riferimento trovato nel localStorage (overlay o metadata mancanti)');
                this.referenceLoaded = false;
            }

        } catch (error) {
            console.warn('Errore nel verificare lo stato del riferimento:', error);
            this.referenceLoaded = false;
            return false;
        }
    }

    showReferenceRequiredMessage() {
        // Questa funzione è gestita da validation.html
    }

    displayReferencePreview(imageData, metadata) {
        // Questa funzione è gestita da validation.html
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
                this.sessionId = 'session_' + Math.random().toString(36).substring(2, 9);
                localStorage.setItem('sensor_validation_session', this.sessionId);
                console.log('Nuovo Session ID generato:', this.sessionId);
            }
        }
    }

    /**
     * Calcola la risoluzione ottimale basata sul container e sull'immagine di riferimento
     */
    calculateOptimalResolution(containerWidth, containerHeight, referenceAspect = null) {
        // Se abbiamo un aspect ratio di riferimento, usalo
        const targetAspect = referenceAspect || (containerWidth / containerHeight);

        // Risoluzione massima supportata dalla maggior parte delle webcam
        const MAX_WIDTH = 1920;
        const MAX_HEIGHT = 1080;

        let width, height;

        if (targetAspect > 1) {
            // Landscape
            width = Math.min(containerWidth * window.devicePixelRatio, MAX_WIDTH);
            height = Math.round(width / targetAspect);
        } else {
            // Portrait
            height = Math.min(containerHeight * window.devicePixelRatio, MAX_HEIGHT);
            width = Math.round(height * targetAspect);
        }

        // Assicura che le dimensioni siano pari (requisito di molti codec)
        width = Math.floor(width / 2) * 2;
        height = Math.floor(height / 2) * 2;

        return {width, height};
    }

    /**
     * Inizializza l'accesso alla camera con risoluzione adattiva
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
            console.log(`Using reference aspect ratio: ${referenceAspect}`);
        } else {
            console.log('No reference image for aspect ratio, using container.');
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
                    deviceId: {exact: preferredDeviceId},
                    width: {ideal: optimalRes.width},
                    height: {ideal: optimalRes.height},
                    aspectRatio: {ideal: referenceAspect || (optimalRes.width / optimalRes.height)}
                }
            });
        }

        // 2. Constraint con risoluzione specifica (senza deviceId)
        if (!isSafari) {
            tryConstraintsList.push({
                video: {
                    width: {ideal: optimalRes.width},
                    height: {ideal: optimalRes.height},
                    aspectRatio: {ideal: referenceAspect || (optimalRes.width / optimalRes.height)},
                    facingMode: 'user'
                }
            });

            // 3. Constraint con risoluzione min/max
            tryConstraintsList.push({
                video: {
                    width: {min: 640, ideal: optimalRes.width, max: 1920},
                    height: {min: 480, ideal: optimalRes.height, max: 1080},
                    facingMode: 'user'
                }
            });
        }

        // 4. Fallback base per Safari
        tryConstraintsList.push({video: true});

        // Enumera dispositivi disponibili
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const cameras = devices.filter(d => d.kind === 'videoinput');

            if (cameras.length > 1 && !preferredDeviceId) {
                console.log(`${cameras.length} camere trovate`);
                // Prova con la prima camera disponibile
                tryConstraintsList.unshift({
                    video: {
                        deviceId: {exact: cameras[0].deviceId},
                        width: {ideal: optimalRes.width},
                        height: {ideal: optimalRes.height}
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
                if (!this.videoElement) {
                    console.error('Elemento video non trovato');
                    stream.getTracks().forEach(t => t.stop());
                    return false;
                }

                this.videoElement.srcObject = stream;
                this.stream = stream;

                // Aspetta il caricamento dei metadata
                await new Promise((resolve) => {
                    const onLoaded = () => {
                        this.videoElement.removeEventListener('loadedmetadata', onLoaded);
                        resolve();
                    };
                    this.videoElement.addEventListener('loadedmetadata', onLoaded);
                    setTimeout(resolve, 2500); // Timeout di sicurezza
                });

                console.log('✅ Camera inizializzata con risoluzione adattiva');

                // Aggiorna badge
                const camStatus = document.getElementById('camStatus');
                if (camStatus) {
                    camStatus.textContent = `Attiva (${settings.width}x${settings.height})`;
                    camStatus.className = 'badge bg-success';
                }

                this.showStatus('Camera pronta per la validazione', 'success');
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

// --- MODIFICA 2: Inizializzazione globale ---
// Rimosso il wrapper 'DOMContentLoaded' per eseguire questo
// script immediatamente non appena viene caricato.
try {
    console.log('🚀 Inizializzazione SensorValidationApp (immediata)...');
    const app = new SensorValidationApp();
    window.sensorApp = app;

    // Inizializzazione camera automatica dopo un breve delay
    setTimeout(() => {
        console.log('Tentativo di inizializzazione camera...');

        // --- MODIFICA 2b: Correggiamo la chiave del localStorage ---
        // 'validation.html' salva l'overlay (per l'aspect ratio)
        // con la chiave 'validationOverlayImage'.
        const storedReference = localStorage.getItem('validationOverlayImage');
        let referenceImage = null;

        if (storedReference) {
            referenceImage = new Image();
            referenceImage.src = storedReference;
            console.log('Trovata immagine "validationOverlayImage" per calcolo aspect ratio');
        } else {
            console.warn('Nessuna immagine "validationOverlayImage" trovata, uso aspect ratio di default');
        }

        // Aspetta che l'immagine (se trovata) sia caricata per leggerne le dimensioni
        const initializeCam = (img) => {
            app.initializeCameraWithAdaptiveResolution(null, img).then(success => {
                if (success) {
                    console.log('✅ Camera initialized successfully');
                } else {
                    console.error('❌ Failed to initialize camera');
                    app.showStatus('Errore camera - verifica i permessi e ricarica la pagina', 'error');
                }
            }).catch(error => {
                console.error('❌ Camera initialization error:', error);
                app.showStatus('Errore inizializzazione camera', 'error');
            });
        };

        if (referenceImage) {
            referenceImage.onload = () => initializeCam(referenceImage);
            referenceImage.onerror = () => initializeCam(null); // Fallback
        } else {
            initializeCam(null); // Avvia subito senza immagine
        }

    }, 500);

} catch (error) {
    console.error('❌ Failed to initialize app:', error);
    const statusElement = document.getElementById('statusMessage');
    if (statusElement) {
        statusElement.textContent = 'Errore di inizializzazione dell\'applicazione';
        statusElement.className = 'ms-3 text-error';
    }
}