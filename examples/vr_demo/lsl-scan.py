from pylsl import resolve_streams
import time

print("Looking for streams...")
# Check available streams
while True:
    streams = resolve_streams()
    print(f"Found {len(streams)} streams:")
    for stream in streams:
        print(f"Stream: {stream.name()}, Type: {stream.type()}, Source ID: {stream.source_id()}")
    time.sleep(1)