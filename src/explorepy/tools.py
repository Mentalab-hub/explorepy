from .parser import Parser
import os.path
import csv


def bin2csv(bin_file):
    filename, extension = os.path.splitext(bin_file)
    assert os.path.isfile(bin_file), "Error: File does not exist!"
    assert extension == '.BIN', "File type error! File extension must be BIN."
    eeg_out_file = filename + '_eeg.csv'
    orn_out_file = filename + '_orn.csv'
    c = 'y'

    if os.path.isfile(eeg_out_file):
        c = input("A file with this name already exist, are you sure you want to proceed? [Enter y/n]")
    while True:
        if c == 'n':
            exit()
        elif c == 'y':
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
                        return 0
        else:
            c = input("A file with this name already exist, are you sure you want to proceed? [Enter y/n]")
