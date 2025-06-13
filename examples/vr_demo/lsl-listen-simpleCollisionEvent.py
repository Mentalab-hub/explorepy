from pylsl import resolve_streams, StreamInlet

# Resolve the stream named "LSL4Unity.Samples.SimpleCollisionEvent"
print("Looking for stream named 'LSL4Unity.Samples.SimpleCollisionEvent'...")
streams = resolve_streams()
unity_streams = [stream for stream in streams if stream.name() == 'LSL4Unity.Samples.SimpleCollisionEvent']

if not unity_streams:
    print("No LSL4Unity.Samples.SimpleCollisionEvent stream found!")
    exit()

# Create an inlet to receive samples
inlet = StreamInlet(unity_streams[0])
print("Connected to LSL4Unity.Samples.SimpleCollisionEvent stream.")

while True:
    sample, timestamp = inlet.pull_sample()
    print(f"Received: {sample[0]} at {timestamp}")