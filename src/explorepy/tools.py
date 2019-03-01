from explorepy.parser import Parser
import os.path
import csv
import bluetooth


def bt_scan():
    print("Searching for nearby devices...")
    nearby_devices = bluetooth.discover_devices(lookup_names=True)

    for address, name in nearby_devices:
        if "Explore" in name:
            print("Device found: %s - %s" % (name, address))

    if len(nearby_devices) == 0:
        print("No Devices found")
        return


def bin2csv(bin_file, do_overwrite, out_dir=None):

    r"""Binary to CSV file converter.
    This function converts the given binary file to ExG and ORN csv files.

    Args:
        bin_file (str): Binary file full address
        out_dir (str): Output directory (if None, uses the same directory as binary file)
        do_overwrite (bool): Overwrite if files exist already

    Returns:

    """
    head_path, full_filename = os.path.split(bin_file)
    filename, extension = os.path.splitext(full_filename)
    assert os.path.isfile(bin_file), "Error: File does not exist!"
    assert extension == '.BIN', "File type error! File extension must be BIN."
    if out_dir is None:
        out_dir = head_path

    eeg_out_file = out_dir + filename + '_eeg.csv'
    orn_out_file = out_dir + filename + '_orn.csv'
    assert not (os.path.isfile(eeg_out_file) and do_overwrite), eeg_out_file + " already exists!"
    assert not (os.path.isfile(orn_out_file) and do_overwrite), orn_out_file + " already exists!"

    with open(bin_file, "rb") as f_bin, open(eeg_out_file, "w") as f_eeg, open(orn_out_file, "w") as f_orn:
        parser = Parser(fid=f_bin)
        f_orn.write('TimeStamp, ax, ay, az, gx, gy, gz, mx, my, mz \n')
        f_orn.write('hh:mm:ss, mg/LSB, mg/LSB, mg/LSB, mdps/LSB, mdps/LSB, mdps/LSB, mgauss/LSB, mgauss/LSB, mgauss/LSB\n')
        f_eeg.write('TimeStamp, ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8\n')
        csv_eeg = csv.writer(f_eeg, delimiter=',')
        csv_orn = csv.writer(f_orn, delimiter=',')
        print("Converting...")
        while True:
            try:
                packet = parser.parse_packet(mode='record', csv_files=(csv_eeg, csv_orn))
            except ValueError:
                print("Binary file ended suddenly! Conversion finished!")
                break


