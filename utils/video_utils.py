import cv2
import numpy as np
from typing import Generator


def generate_frames(video_source: int = 0) -> Generator[bytes, None, None]:
    """Generate video frames from camera"""
    camera = cv2.VideoCapture(video_source)

    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        camera.release()


def draw_validation_results(image: np.ndarray, validation_results: dict, current_points: dict = None) -> np.ndarray:
    result_image = image.copy()

    # Draw limb points if available
    if current_points:
        for point_name, (x, y) in current_points.items():
            cv2.circle(result_image, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(result_image, point_name, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Draw sensor validation results
    for sensor_result in validation_results.get('sensor_results', []):
        if sensor_result.get('matched', False):
            actual_pos = sensor_result['actual_position']
            expected_pos = sensor_result['expected_position']

            # Draw actual position
            color = (0, 255, 0) if sensor_result['correctly_placed'] else (0, 0, 255)

            cv2.circle(result_image, actual_pos, 8, color, 2)

            # Draw expected position
            cv2.circle(result_image, expected_pos, 4, (255, 255, 0), -1)

            # Draw correction vector if needed
            if not sensor_result['correctly_placed']:
                correction = sensor_result['correction_vector']
                end_point = (
                    int(actual_pos[0] + correction[0]),
                    int(actual_pos[1] + correction[1])
                )
                cv2.arrowedLine(result_image, actual_pos, end_point, (0, 0, 255), 2)

                # Draw distance text
                distance = sensor_result['position_error']
                cv2.putText(result_image, f"{distance:.1f}px",
                            (actual_pos[0] + 10, actual_pos[1] + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw overall score
    overall_score = validation_results.get('overall_score', 0)
    score_color = (0, 255, 0) if overall_score > 80 else (0, 255, 255) if overall_score > 60 else (0, 0, 255)

    cv2.putText(result_image, f'Score: {overall_score: .1f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)

    # Draw status
    status = "CORRETTO" if overall_score > 80 else "DA REGOLARE" if overall_score > 60 else "ERRATO"
    cv2.putText(result_image, f'Stato: {status}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

    return result_image
