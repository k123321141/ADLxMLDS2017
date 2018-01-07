
import argparse, re


def main(path):
    with open(path, 'r') as f:
        buf = f.read()

    pg = re.compile('generator \(train\).*\d\.\d.*\d\.\d.*\d\.\d.*\d\.\d.*')
    pd = re.compile('discriminator (train).*\d\.\d.*\d\.\d.*\d\.\d.*\d\.\d.*')
    value = re.compile('\d\.\d')
    m1 = pg.findall(buf)
    for m in m1:
        mm = value.findall(m)
        assert len(mm) == 4
        print mm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot log')
    parser.add_argument('log',default=None, help='log file')
    args = parser.parse_args()
    main(args.log)
