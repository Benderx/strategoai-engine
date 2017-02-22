import pandas
import numpy as np


CSV_LOCATION = 'games'

def import_data():
    return pandas.read_pickle(CSV_LOCATION)


def shuffle(df):     
    print("Shuffling data")
    df = df.sample(frac=1).reset_index(drop=True)
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
    # np.random.seed(seed=4567)

    data = import_data()

    # counter = 0
    # for index, row in data.iterrows():
    #     if counter > 0:
    #         break
    #     print('row num', index)
    #     print_board(row['board'])
    #     # print(row['move_data'])
    #     print(row['move_from'] % 6, int(row['move_from']/6))
    #     print(row['move_to'] % 6, int(row['move_to']/6))
    #     counter += 1

    sdata = shuffle(data)

    counter = 0
    for index, row in sdata.iterrows():
        if counter > 0:
            break
        print('row num', index)
        print_board(row['board'])
        # print(row['move_data'])
        print(row['move_from'] % 6, int(row['move_from']/6))
        print(row['move_to'] % 6, int(row['move_to']/6))
        counter += 1

main()