import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

def main():
    info = StreamInfo('LSLExampleAmp', 'Scale', 3, 100, 'float32', 'unity_scale_stream')
    
    outlet = StreamOutlet(info)
    
    print("Streaming started. Press Ctrl+C to stop.")
    
    try:
        while True:
            sample = np.random.rand(3)  # 3 random values between 0 and 1
            outlet.push_sample(sample)
            time.sleep(0.01)  # 100 Hz sampling rate
            
    except KeyboardInterrupt:
        print("Streaming stopped.")

if __name__ == '__main__':
    main()
