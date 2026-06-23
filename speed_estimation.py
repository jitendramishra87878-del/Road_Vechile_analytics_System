def estimate_speed(distance_meters=50.0, time_seconds=4.0):
    if time_seconds <= 0:
        raise ValueError("time_seconds must be greater than zero")

    speed_mps = distance_meters / time_seconds
    return speed_mps * 3.6


def main():
    speed_kmph = estimate_speed(distance_meters=50, time_seconds=4)
    print(f"Speed = {speed_kmph:.2f} km/h")


if __name__ == "__main__":
    main()