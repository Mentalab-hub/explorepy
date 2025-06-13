# VR Demo with LSL Integration

This example demonstrates how to integrate Lab Streaming Layer (LSL) with a VR application using Unity.

## Prerequisites

1. Unity (latest LTS version recommended)
2. Python 3.x
3. Required Python packages:
   - pylsl
   - numpy

## Unity Setup

1. Create a new Unity project
2. Import the LSL4Unity package from the Unity Package manager
    Open the Package Manager Window, click on the `+` dropdown, and choose `Add package from git URL....` Enter the followingURL: `https://github.com/labstreaminglayer/LSL4Unity.git`

## Python Setup

1. Install required Python packages:
   ```bash
   pip install pylsl numpy
   ```

## Running the Demo

1. Start streaming with lsl:
   ```bash
   python lsl-stream-simpleInlentExample.py
   ```

2. Start Unity:
   - Open the project in Unity
   - In the project tab - navigate to `/Assets/Samples/labstreaminglayer for Unity/1.16.0/SimpleInletScaleObject` and open the scene
   - Press Play in the Unity Editor

`lsl-stream-simpleInlentExample` will stream random 3D coordinates through lsl and unity example will listen and change the size of the displayed cylinder accordingly.

## Implementation Details

The implementation consists of three Python scripts that demonstrate LSL communication:

1. `lsl-scan.py`: Utility script to discover available LSL streams
2. `lsl-stream-unity.py`: Example script that streams random data
3. `lsl-listen-simpleCollisionEvent.py`: Example script that listens to collision events from Unity Sample simpleCollisionEvent. 

## Notes

- The current implementation uses the sample scenes provided by LSL4Unity
- The Python scripts serve as examples for LSL communication

## Resources

- [LSL4Unity Documentation](https://github.com/labstreaminglayer/LSL4Unity)
