from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
import json
import base64
import numpy as np
import cv2
import logging

from config import Config
from services.validation_service import ValidationService
from utils.image_processing import encode_image_to_base64
from utils.background_removal import remove_background_from_pil
from PIL import Image
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.error(f"Errore nel verificare il riferimento: {str(e)}")
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
    Endpoint per caricare un'immagine di riferimento CON elaborazione landmark
    Usa ValidationService.handle_reference_upload che include:
    - Salvataggio immagine
    - Detection landmark (se limb_type='leg')
    - Annotazione immagine
    """
    try:
        logger.info("[ROUTE] /api/upload_reference called - WITH LANDMARK DETECTION")

        # Usa il metodo del ValidationService che abbiamo modificato
        result = validation_service.handle_reference_upload(request)

        if result.get('success'):
            logger.info("✅ Upload e processing completato con successo")

            # Log landmark info se presenti
            if 'landmarks' in result:
                landmarks_info = result['landmarks']
                logger.info(f"   📍 Landmark rilevati: {landmarks_info.get('total_landmarks', 0)}")
                logger.info(f"   🔬 Metodo: {landmarks_info.get('detection_method', 'N/A')}")

                if 'sensor_positions' in landmarks_info:
                    logger.info(f"   💉 Posizioni sensori calcolate: {len(landmarks_info['sensor_positions'])}")
        else:
            logger.warning(f"⚠️ Upload fallito: {result.get('error', 'Unknown error')}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"[ERROR] Route /api/upload_reference failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Errore server: {str(e)}'
        }), 500


@app.route('/api/remove_background', methods=['POST'])
def api_remove_background():
    """
    Endpoint per rimuovere lo sfondo da un'immagine
    """
    try:
        logger.info("[ROUTE] /api/remove_background called")
        data = request.get_json()

        if not data or 'image_base64' not in data:
            return jsonify({'success': False, 'error': 'Parametro image_base64 mancante'}), 400

        image_base64 = data.get('image_base64')
        use_preprocess = data.get('use_preprocess', True)
        resize_max = data.get('resize_max', 1024)
        grabcut_if_failed = data.get('grabcut_if_failed', True)

        # Decodifica immagine
        try:
            if isinstance(image_base64, str) and ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]
            image_bytes = base64.b64decode(image_base64)
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception:
            return jsonify({'success': False, 'error': 'Immagine non valida'}), 400

        # Rimozione sfondo
        out_pil, method = remove_background_from_pil(
            pil_img,
            use_preprocess=use_preprocess,
            resize_max=resize_max,
            grabcut_if_failed=grabcut_if_failed
        )

        if out_pil is not None:
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            result_image = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            logger.info(f"✅ Sfondo rimosso con successo usando {method}")

            return jsonify({
                'success': True,
                'image_base64': result_image,
                'method': method,
                'message': 'Sfondo rimosso con successo'
            })
        else:
            logger.warning("⚠️ Rimozione sfondo fallita")
            return jsonify({'success': False, 'error': 'Impossibile rimuovere lo sfondo'}), 500

    except Exception as e:
        logger.error(f"[ERROR] Route /api/remove_background failed: {e}")
        return jsonify({'success': False, 'error': f'Errore server: {str(e)}'}), 500



@app.route('/api/validate_frame', methods=['POST'])
def api_validate_frame():
    """
    Endpoint per validare un frame dalla webcam
    Riceve il frame corrente e l'immagine di riferimento in base64
    """
    try:
        logger.info("[ROUTE] /api/validate_frame called")
        result = validation_service.validate_current_frame(request)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[ERROR] Route /api/validate_frame failed: {e}")
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
        logger.info(f"[ROUTE] /api/get_validation_status called for session: {session_id}")
        result = validation_service.get_validation_status(session_id)
        return jsonify(result)
    except Exception as e:
        logger.error(f"[ERROR] Route /api/get_validation_status failed: {e}")
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
        logger.error(f"[ERROR] Route /api/check_reference failed: {e}")
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
    logger.error(f"[ERROR 500] {error}")
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
    print("   • POST /api/upload_reference        - Carica immagine riferimento")
    print("   • POST /api/remove_background       - Rimuove sfondo da immagine")
    print("   • POST /api/validate_frame          - Valida frame corrente")
    print("   • GET  /api/get_validation_status   - Stato della validazione")
    print("   • POST /api/check_reference         - Verifica esistenza riferimento")

    port = int(os.environ.get('PORT', 5000))
    print("\n🚀 Avvio applicazione Flask...")
    print(f"   URL: http://localhost:{port}")
    print(f"   URL locale: http://127.0.0.1:{port}")
    print("\n" + "=" * 50)
    print("Premi CTRL+C per fermare il server")
    print("=" * 50 + "\n")

    app.run(debug=True, host='127.0.0.0', port=port)