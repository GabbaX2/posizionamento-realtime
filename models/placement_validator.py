import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from scipy.spatial import distance
import cv2


class PlacementValidator:
    def __init__(self, reference_image_path: Optional[str] = None, limb_type: str = 'arm'):
        self.limb_type = limb_type
        self.reference_data = None
        self.position_tolerance = 25  # pixels
        self.angle_tolerance = 15  # degrees
        self.last_error = None  # Store last error for debugging

        print(f"\n🔧 Inizializzazione PlacementValidator per '{limb_type}'...")

        if reference_image_path and os.path.exists(reference_image_path):
            success = self.load_reference(reference_image_path)
            if not success:
                print(f"❌ ERRORE: Impossibile caricare il riferimento")
                print(f"   Motivo: {self.last_error}")
                # Reference_data rimane None
            else:
                print(f"✅ PlacementValidator inizializzato con successo")
        else:
            if reference_image_path:
                print(f"❌ File di riferimento non trovato: {reference_image_path}")
            else:
                print(f"ℹ️  Nessun file di riferimento fornito")

    def load_reference(self, image_path: str) -> bool:
        """Load and process reference image"""
        print(f"\n📂 Caricamento riferimento da: {image_path}")
        
        try:
            from models.limb_detector import LimbDetector
            from models.sensor_detector import SensorDetector

            # Verify file exists
            if not os.path.exists(image_path):
                self.last_error = f"File non trovato: {image_path}"
                print(f"❌ {self.last_error}")
                return False

            # Load image
            print(f"   📸 Caricamento immagine...")
            image = cv2.imread(image_path)
            if image is None:
                self.last_error = "Impossibile leggere l'immagine (formato non supportato o file corrotto)"
                print(f"❌ {self.last_error}")
                return False

            print(f"   ✅ Immagine caricata: {image.shape}")

            # Initialize detectors
            print(f"   🔧 Inizializzazione detector...")
            limb_detector = LimbDetector(self.limb_type)
            sensor_detector = SensorDetector()

            # Extract reference points
            print(f"   🔍 Rilevamento punti anatomici...")
            reference_points = limb_detector.get_limb_points(image)
            
            if reference_points is None:
                self.last_error = f"Arto '{self.limb_type}' non rilevato nell'immagine di riferimento. Assicurati che l'arto sia ben visibile e completamente nell'inquadratura."
                print(f"❌ {self.last_error}")
                return False

            print(f"   ✅ Punti anatomici rilevati: {len(reference_points)}")
            for name, pos in reference_points.items():
                print(f"      • {name}: {pos}")

            # Extract reference sensors
            print(f"   🎨 Rilevamento sensori colorati...")
            reference_sensors = sensor_detector.detect_sensors(image)
            
            if not reference_sensors:
                self.last_error = "Nessun sensore colorato rilevato nell'immagine di riferimento. Verifica che i sensori siano ben visibili con colori distinti (rosso, verde, blu, giallo)."
                print(f"❌ {self.last_error}")
                return False

            print(f"   ✅ Sensori rilevati: {len(reference_sensors)}")
            for i, sensor in enumerate(reference_sensors):
                print(f"      • Sensore {i+1}: colore={sensor['color']}, posizione={sensor['position']}, area={sensor['area']:.0f}")

            # Calculate relative positions and relationships
            print(f"   📐 Calcolo posizioni relative...")
            relative_positions = self._calculate_relative_positions(reference_points, reference_sensors)
            
            print(f"   ✅ Posizioni relative calcolate: {len(relative_positions)}")
            for i, rel_pos in enumerate(relative_positions):
                print(f"      • Sensore {i+1} ({rel_pos['sensor_data']['color']}): punto più vicino = {rel_pos['closest_point']}, offset = {rel_pos['relative_vector']}")

            # Calculate limb orientation
            limb_orientation = limb_detector.get_limb_orientation(reference_points)
            print(f"   📏 Orientamento arto: {limb_orientation:.1f}°")

            # Store reference data
            self.reference_data = {
                'points': reference_points,
                'sensors': reference_sensors,
                'image_shape': image.shape,
                'relative_positions': relative_positions,
                'limb_orientation': limb_orientation
            }

            # Save reference data for persistence
            print(f"   💾 Salvataggio dati di riferimento...")
            self._save_reference_data(image_path)

            print(f"✅ Riferimento caricato con successo!")
            print(f"   • Punti anatomici: {len(reference_points)}")
            print(f"   • Sensori: {len(reference_sensors)}")
            print(f"   • Orientamento: {limb_orientation:.1f}°")

            self.last_error = None
            return True

        except Exception as e:
            self.last_error = f"Errore inaspettato: {type(e).__name__}: {str(e)}"
            print(f"❌ {self.last_error}")
            import traceback
            traceback.print_exc()
            return False

    def _calculate_relative_positions(self, points: Dict, sensors: List[Dict]) -> List[Dict]:
        """Calculate relative positions of sensors to limb points"""
        relative_positions = []

        for sensor in sensors:
            sensor_pos = sensor['position']

            # Calculate distances to all limb points
            distances = {}
            for point_name, point_pos in points.items():
                dist = distance.euclidean(sensor_pos, point_pos)
                distances[point_name] = dist

            # Find closest limb point
            closest_point = min(distances.items(), key=lambda x: x[1])[0]

            # Calculate relative position vector
            closest_point_pos = points[closest_point]
            relative_vector = (
                sensor_pos[0] - closest_point_pos[0],
                sensor_pos[1] - closest_point_pos[1]
            )

            relative_positions.append({
                'sensor_data': sensor,
                'closest_point': closest_point,
                'relative_vector': relative_vector,
                'distances': distances
            })

        return relative_positions

    def validate_current_placement(self, current_points: Dict, current_sensors: List[Dict]) -> Dict:
        """Validate current sensor placement against reference"""
        if self.reference_data is None:
            return {
                'error': 'No reference data loaded',
                'overall_score': 0,
                'sensor_results': [],
                'limb_detected': False
            }

        validation_results = {
            'overall_score': 0,
            'sensor_results': [],
            'limb_detected': current_points is not None,
            'sensors_detected': len(current_sensors)
        }

        if current_points is None:
            validation_results['error'] = 'Limb not detected in current frame'
            return validation_results

        # Validate each sensor
        total_score = 0
        valid_sensors = 0

        for ref_sensor_data in self.reference_data['relative_positions']:
            sensor_validation = self._validate_sensor_position(
                ref_sensor_data, current_points, current_sensors
            )
            validation_results['sensor_results'].append(sensor_validation)

            if sensor_validation['matched']:
                total_score += sensor_validation['position_score']
                valid_sensors += 1

        # Calculate overall score
        if valid_sensors > 0:
            validation_results['overall_score'] = total_score / valid_sensors
        else:
            validation_results['overall_score'] = 0

        # Check limb orientation
        from models.limb_detector import LimbDetector
        limb_detector = LimbDetector(self.limb_type)
        current_orientation = limb_detector.get_limb_orientation(current_points)
        orientation_diff = abs(current_orientation - self.reference_data['limb_orientation'])

        validation_results['orientation_correct'] = orientation_diff <= self.angle_tolerance
        validation_results['orientation_difference'] = orientation_diff

        return validation_results

    def _validate_sensor_position(self, ref_sensor_data: Dict, current_points: Dict,
                                  current_sensors: List[Dict]) -> Dict:
        """Validate position of a single sensor"""
        # Find matching sensor in current frame
        matched_sensor = self._find_matching_sensor(ref_sensor_data, current_sensors)

        if matched_sensor is None:
            return {
                'matched': False,
                'sensor_color': ref_sensor_data['sensor_data']['color'],
                'error': f"Sensore {ref_sensor_data['sensor_data']['color']} non trovato nel frame corrente",
                'position_score': 0,
                'position_error': 0,
                'expected_position': (0, 0),
                'actual_position': (0, 0),
                'correction_vector': None,
                'correctly_placed': False
            }

        # Calculate expected position
        expected_position = self._calculate_expected_position(ref_sensor_data, current_points)
        actual_position = matched_sensor['position']

        # Calculate position error
        position_error = distance.euclidean(expected_position, actual_position)
        position_score = max(0, 100 - (position_error / self.position_tolerance) * 100)

        # Calculate correction vector
        correction_vector = (
            expected_position[0] - actual_position[0],
            expected_position[1] - actual_position[1]
        )

        return {
            'matched': True,
            'sensor_color': ref_sensor_data['sensor_data']['color'],
            'expected_position': expected_position,
            'actual_position': actual_position,
            'position_error': position_error,
            'position_score': position_score,
            'correction_vector': correction_vector,
            'correctly_placed': position_error <= self.position_tolerance
        }

    def _find_matching_sensor(self, ref_sensor_data: Dict, current_sensors: List[Dict]) -> Optional[Dict]:
        """Find sensor in current frame that matches reference sensor"""
        ref_color = ref_sensor_data['sensor_data']['color']
        ref_area = ref_sensor_data['sensor_data']['area']

        # Find sensors of same color
        same_color_sensors = [s for s in current_sensors if s['color'] == ref_color]

        if not same_color_sensors:
            return None

        # Find sensor with most similar area
        area_differences = [abs(s['area'] - ref_area) for s in same_color_sensors]
        best_match_idx = np.argmin(area_differences)

        return same_color_sensors[best_match_idx]

    def _calculate_expected_position(self, ref_sensor_data: Dict, current_points: Dict) -> Tuple[int, int]:
        """Calculate expected sensor position based on current limb points"""
        closest_point_name = ref_sensor_data['closest_point']
        relative_vector = ref_sensor_data['relative_vector']

        if closest_point_name not in current_points:
            # Fallback: use first available point
            print(f"⚠️ Punto '{closest_point_name}' non trovato, uso fallback")
            closest_point_name = list(current_points.keys())[0]

        current_point_pos = current_points[closest_point_name]

        expected_x = current_point_pos[0] + relative_vector[0]
        expected_y = current_point_pos[1] + relative_vector[1]

        return (int(expected_x), int(expected_y))

    def _save_reference_data(self, image_path: str) -> None:
        """Save reference data to JSON file for persistence"""
        try:
            # Ensure directory exists
            ref_dir = "data/reference_positions"
            os.makedirs(ref_dir, exist_ok=True)

            # Create filename based on image path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            ref_data_path = os.path.join(ref_dir, f"{base_name}_reference.json")

            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {
                'limb_type': self.limb_type,
                'points': {k: list(v) for k, v in self.reference_data['points'].items()},
                'sensors': [
                    {
                        'position': list(sensor['position']),
                        'color': sensor['color'],
                        'area': float(sensor['area']),
                        'bbox': list(sensor['bbox'])
                    }
                    for sensor in self.reference_data['sensors']
                ],
                'image_shape': list(self.reference_data['image_shape']),
                'relative_positions': [
                    {
                        'closest_point': rp['closest_point'],
                        'relative_vector': list(rp['relative_vector']),
                        'distances': {k: float(v) for k, v in rp['distances'].items()},
                        'sensor_data': {
                            'color': rp['sensor_data']['color'],
                            'area': float(rp['sensor_data']['area'])
                        }
                    }
                    for rp in self.reference_data['relative_positions']
                ],
                'limb_orientation': float(self.reference_data['limb_orientation'])
            }

            with open(ref_data_path, 'w') as f:
                json.dump(serializable_data, f, indent=2)

            print(f"   ✅ Dati salvati in: {ref_data_path}")

        except Exception as e:
            print(f"   ⚠️ Errore nel salvataggio dati di riferimento: {e}")
            import traceback
            traceback.print_exc()

    def load_reference_from_json(self, json_path: str) -> bool:
        """Load reference data from JSON file"""
        try:
            print(f"📂 Caricamento riferimento da JSON: {json_path}")
            
            with open(json_path, 'r') as f:
                data = json.load(f)

            self.limb_type = data['limb_type']
            self.reference_data = {
                'points': {k: tuple(v) for k, v in data['points'].items()},
                'sensors': data['sensors'],
                'image_shape': tuple(data['image_shape']),
                'relative_positions': data['relative_positions'],
                'limb_orientation': data['limb_orientation']
            }

            print(f"✅ Riferimento caricato da JSON con successo")
            self.last_error = None
            return True

        except Exception as e:
            self.last_error = f"Errore nel caricamento da JSON: {str(e)}"
            print(f"❌ {self.last_error}")
            import traceback
            traceback.print_exc()
            return False