from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
import json
import base64
import numpy as np
import cv2

from config import Config
from services.validation_service import ValidationService
from utils.image_processing import encode_image_to_base64

app = Flask(__name__)
app.config.from_object(Config)

# Inizializza il servizio di validazione
validation_service = ValidationService()


def check_reference_exists(session_id):
    """Verifica se esiste un riferimento per la sessione specificata"""
    try:
        reference_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_reference.json")

        if os.path.exists(reference_path):
            with open(reference_path, 'r') as f:
                reference_data = json.load(f)

            return {
                'success': True,
                'has_reference': True,
                'preview_image': reference_data.get('preview_image'),
                'limb_type': reference_data.get('limb_type'),
                'sensor_count': len(reference_data.get('sensor_positions', [])),
                'message': 'Riferimento trovato'
            }
        else:
            return {
                'success': True,
                'has_reference': False,
                'message': 'Nessun riferimento trovato'
            }

    except Exception as e:
        print(f"Errore nel verificare il riferimento: {str(e)}")
        return {
            'success': False,
            'has_reference': False,
            'error': str(e)
        }


# ============================================================================
# ROUTES - PAGINE HTML
# ============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    return render_template('upload.html')


@app.route('/validation')
def validation_page():
    return render_template('validation.html')


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/upload_reference', methods=['POST'])
def api_upload_reference():
    """
    Endpoint per caricare un'immagine di riferimento SENZA elaborazione
    Salva solo l'immagine originale
    """
    try:
        print("[ROUTE] /api/upload_reference called - SIMPLE UPLOAD (no detection)")

        # Verifica presenza file
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Nessun file fornito'
            }), 400

        file = request.files['file']
        limb_type = request.form.get('limb_type', 'arm')

        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Nessun file selezionato'
            }), 400

        # Leggi l'immagine
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                'success': False,
                'error': 'Impossibile decodificare l\'immagine'
            }), 400

        # Converti in base64 per il frontend
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Salva il file
        session_id = request.form.get('session_id', 'default')
        upload_folder = app.config.get('UPLOAD_FOLDER', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        # Salva immagine originale
        image_path = os.path.join(upload_folder, f"{session_id}_reference.jpg")
        cv2.imwrite(image_path, image)

        # Salva metadata JSON
        metadata = {
            'limb_type': limb_type,
            'filename': file.filename,
            'upload_date': str(np.datetime64('now')),
            'image_path': image_path,
            'preview_image': image_base64
        }

        metadata_path = os.path.join(upload_folder, f"{session_id}_reference.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

        print(f"✅ Immagine salvata: {file.filename}, Tipo: {limb_type}")
        print(f"   Path: {image_path}")

        return jsonify({
            'success': True,
            'message': 'Immagine caricata con successo (senza elaborazione)',
            'preview_image': image_base64,
            'limb_type': limb_type,
            'filename': file.filename,
            'session_id': session_id
        })

    except Exception as e:
        print(f"[ERROR] Route /api/upload_reference failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Errore server: {str(e)}'
        }), 500


@app.route('/api/validate_frame', methods=['POST'])
def api_validate_frame():
    """
    Endpoint per validare un frame dalla webcam
    Riceve il frame corrente e l'immagine di riferimento in base64
    """
    try:
        print("[ROUTE] /api/validate_frame called")
        result = validation_service.validate_current_frame(request)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Route /api/validate_frame failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Errore server: {str(e)}'
        }), 500


@app.route('/api/get_validation_status', methods=['GET'])
def api_get_validation_status():
    """
    Endpoint per ottenere lo stato della validazione per una sessione
    """
    try:
        session_id = request.args.get('session_id', 'default')
        print(f"[ROUTE] /api/get_validation_status called for session: {session_id}")
        result = validation_service.get_validation_status(session_id)
        return jsonify(result)
    except Exception as e:
        print(f"[ERROR] Route /api/get_validation_status failed: {e}")
        return jsonify({
            'success': False,
            'error': f'Errore server: {str(e)}'
        }), 500


@app.route('/api/check_reference', methods=['POST'])
def api_check_reference():
    """
    Verifica se esiste un riferimento per la sessione corrente
    """
    try:
        data = request.get_json()
        session_id = data.get('session_id', 'default')

        # Verifica l'esistenza del riferimento
        result = check_reference_exists(session_id)
        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Route /api/check_reference failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# FILE SERVING
# ============================================================================

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve file caricati nella cartella uploads"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handler per errori 404"""
    return jsonify({
        'success': False,
        'error': 'Endpoint non trovato'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handler per errori 500"""
    print(f"[ERROR 500] {error}")
    return jsonify({
        'success': False,
        'error': 'Errore interno del server'
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Crea le directory necessarie
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('data/reference_positions', exist_ok=True)
    os.makedirs('data/calibration', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    print("=" * 50)
    print("=== Sensor Placement Validator ===")
    print("=" * 50)
    print("\n📁 Directory create:")
    print("   ✓ uploads/")
    print("   ✓ data/reference_positions/")
    print("   ✓ data/calibration/")
    print("   ✓ static/js/")

    print("\n🔧 Configurazione:")
    print(f"   • Debug mode: {app.config.get('DEBUG', True)}")
    print(f"   • Upload folder: {app.config.get('UPLOAD_FOLDER', 'uploads')}")

    print("\n🌐 API Endpoints disponibili:")
    print("   • POST /api/upload_reference       - Carica immagine riferimento")
    print("   • POST /api/validate_frame         - Valida frame corrente")
    print("   • GET  /api/get_validation_status  - Stato della validazione")
    print("   • POST /api/check_reference        - Verifica esistenza riferimento")

    port = int(os.environ.get('PORT', 6000))
    print("\n🚀 Avvio applicazione Flask...")
    print(f"   URL: http://localhost:{port}")
    print(f"   URL locale: http://127.0.0.1:{port}")
    print("\n" + "=" * 50)
    print("Premi CTRL+C per fermare il server")
    print("=" * 50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=port)