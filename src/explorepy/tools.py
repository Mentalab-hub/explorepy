from .parser import Parser
import os.path
import csv


def bin2csv(bin_file):
    filename, extension = os.path.splitext(bin_file)
    assert extension == '.BIN', "File type error! File extension must be BIN."
    eeg_out_file = filename + '_eeg.csv'
    orn_out_file = filename + '_orn.csv'

    with open(bin_file, "rb") as f_bin, open(eeg_out_file, "w") as f_eeg, open(orn_out_file, "w") as f_orn:
        parser = Parser(fid=f_bin)
        f_orn.write('TimeStamp, ax, ay, az, gx, gy, gz, mx, my, mz \n')
        f_orn.write('hh:mm:ss, mg/LSB, mg/LSB, mg/LSB, mdps/LSB, mdps/LSB, mdps/LSB, mgauss/LSB, mgauss/LSB, mgauss/LSB\n')
        f_eeg.write('TimeStamp, ch1, ch2, ch3, ch4\n')  # , ch5, ch6, ch7, ch8
        csv_eeg = csv.writer(f_eeg, delimiter=',')
        csv_orn = csv.writer(f_orn, delimiter=',')
        pid, timestamp, data = parser.parse_packet(mode='read')
        while pid:
            if pid == 144:
                csv_eeg.writerows(data.T.tolist())
            if pid == 13:
                csv_orn.writerow([timestamp] + data.tolist())

            pid, timestamp, data = parser.parse_packet(mode='read')
