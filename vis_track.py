import fastf1 as f1 
import matplotlib.pyplot as plt
import numpy as np

# Function to rotate a point by a given angle
def rotate(xy, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return xy @ rot_mat


def visualise_track(track, year):
    session = f1.get_session(year, track, "R")
    session.load()

    circuit = session.get_circuit_info()

    pos = session.laps.pick_fastest().get_pos_data()

    # Convert the position data from the fastest lap into an (n, 2) np array
    track = pos[['X', 'Y']].to_numpy()

    # Convert the rotation angle of the circuit from degrees to radian.
    track_angle = (circuit.rotation / 180) * np.pi

    # Rotate the telemetry data of the fastest lap using the track angle, and then plotting it.
    rotated_track = rotate(track, track_angle)
    plt.plot(rotated_track[:, 0], rotated_track[:, 1])

    # Used to offset the position of the corner markers. to make for easier plotting.
    offset_vector = [500, 0]

    for _, corner in circuit.corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"

        # Convert the angle from degrees to radian.
        offset_angle = corner['Angle'] / 180 * np.pi

        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

        # Add the offset to the position of the corner
        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y

        # Rotate the text position equivalently to the rest of the track map
        text_x, text_y = rotate([text_x, text_y], angle=track_angle)

        # Rotate the center of the corner equivalently to the rest of the track map
        track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

        # Draw a circle next to the track.
        plt.scatter(text_x, text_y, color='grey', s=140)

        # Draw a line from the track to this circle.
        plt.plot([track_x, text_x], [track_y, text_y], color='grey')

        # Finally, print the corner number inside the circle.
        plt.text(text_x, text_y, txt,
                va='center_baseline', ha='center', size='small', color='white')

    plt.show()

visualise_track("Silverstone", 2021)