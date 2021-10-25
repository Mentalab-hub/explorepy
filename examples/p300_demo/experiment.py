import argparse
from psychopy import visual, core, event
import random
from explorepy import Explore


def main():
    parser = argparse.ArgumentParser(description="A script to run a visual oddball Experiment")
    parser.add_argument("-n", "--name", dest="name", type=str, help="Name of the device.")
    parser.add_argument("-f", "--filename", dest="filename", type=str, help="Record file name")
    args = parser.parse_args()

    n_blocks = 10   # Number of blocks
    n_trials_per_block = 5   # Number of trials (targets) in each block
    isi = .6    # Inter-stimulus interval
    stim_dur = .5     # Stimulus duration
    labels = [10 for i in range(4)] + [11]  # Stimulus onset labels: 10 -> nontarget and 11 -> target

    # Connect to Explore and record data
    explore = Explore()
    explore.connect(device_name=args.name)
    explore.record_data(file_name=args.filename, file_type='csv', do_overwrite=False)

    # Main window
    win = visual.Window(size=(600, 800), fullscr=True, screen=0, color=[0.1, 0.1, 0.1])

    # A crosshair for fixation of subject's gaze
    fixation = visual.TextStim(win=win, text="+", height=.05, color=(-1, -1, -1))

    # Block start instruction
    wait_text_stim = visual.TextStim(win=win, text="Press space to continue", color=(-1, -1, -1), height=.1, bold=True)

    # Standard and target stimuli
    nontarget = visual.Rect(win, size=.4, fillColor='blue', lineColor=None, opacity=.5)
    target = visual.Circle(win, size=.4, fillColor='red')

    def show_trial(stim_labels):
        for label in stim_labels:
            if label == 10:
                nontarget.draw()
            else:
                target.draw()
            event.clearEvents()
            win.flip()
            explore.set_marker(label)
            clock = core.Clock()
            while clock.getTime() < stim_dur:
                if event.waitKeys(maxWait=.002, keyList=['space'], clearEvents=False):
                    explore.set_marker(20)
                    event.clearEvents()
            win.flip()
            core.wait(isi)

    for b in range(n_blocks):
        wait_text_stim.draw()
        win.flip()
        event.waitKeys(keyList=['space'])
        explore.set_marker(8)
        fixation.draw()
        win.flip()
        core.wait(1)
        for t in range(n_trials_per_block):
            show_trial(labels)
            random.shuffle(labels)

    explore.stop_recording()
    explore.disconnect()

    del wait_text_stim, fixation


if __name__ == '__main__':
    main()

