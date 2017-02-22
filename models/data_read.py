import pandas
import numpy as np


CSV_LOCATION = 'games'

def import_data():
    return pandas.read_pickle(CSV_LOCATION)


def shuffle(df, n=1, axis=0):     
    print("Shuffling data")
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    print("Done shuffling")
    return df


def print_board(board_to_print):
    print()
    for y in range(6):
        arr_temp = []
        for x in range(6):
            val = board_to_print[x + y*6]
            if val == 12:
                name = 'F'
            elif val == 10:
                name = 'B'
            elif val == 11:
                name = 'S'
            elif val == 13:
                name = 'L'
            else:
                name = str(val)
            arr_temp.append(name)
        print(' '.join(map(str, arr_temp)))
    print()


def main():
    data = import_data()

    counter = 0
    for index, row in data.iterrows():
        if counter > 8:
            break
        print('row num', index)
        print_board(row['board'])
        print(row['move_data'])
        print(row['move_from'] % 6, int(row['move_from']/6))
        print(row['move_to'] % 6, int(row['move_to']/6))
        counter += 1

    # sdata = shuffle(data)

    # board = sdata['board'].tolist()
    # owner = sdata['owner'].tolist()
    # move_from = sdata['move_from'].tolist()
    # move_from_one_hot = sdata['move_from_one_hot'].tolist()
    # move_to = sdata['move_to'].tolist()
    # move_to_one_hot = sdata['move_to_one_hot'].tolist()

    # counter = 0
    # for index, row in sdata.iterrows():
    #     if counter > 8:
    #         break
    #     print('row num', index)
    #     print(row['board'])
    #     print(row['move_from'])
    #     print(row['move_to'])
    #     counter += 1

main()