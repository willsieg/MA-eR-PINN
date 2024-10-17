# %%
import pandas as pd
import os

def trip_data_info(relative_path):

    from os import listdir, getcwd
    from os.path import isfile, join, exists

    # data base directory path:
    data_dir = relative_path
    cwd = getcwd()

    if exists(join(cwd,data_dir)):
        data_path = join(cwd,data_dir)
    else:
        print("Directory '",data_dir,"' not found!")
        print("Files and directories in '", cwd, "' :") 
        print(listdir(cwd))
        quit()

    # create list of all parquet files:
    files_list = [f for f in listdir(data_path) if (isfile(join(data_path, f)) and f.endswith(".parquet"))]

    id_num_list,V_list = ([],[])
    trips  = {}

    for f in files_list:
        f = f.strip("v_.parquet")
        x = f.split("_",1)

        id = x[0].split("V")[0].strip("id")
        id_num_list.append(id)

        V = x[0].split("V")[1]
        V_list.append(V)

        trip = x[1].strip("trip")
        if V in trips.keys():
            trips[V].append(trip)
        else:
            trips[V] = [trip]

    vehicles = set(V_list)
    ids = set(id_num_list)

    trip_counts = trips.copy()
    trips_compl = trips.copy()
    for V in trip_counts.keys():
        trips_compl[V] = []
        for t in trip_counts[V]:
            trips_compl[V].append(t.split('_')[0])

        trips_compl[V] = list(set(trips_compl[V]))
        trip_counts[V] = len(trips_compl[V])

        trips_compl[V].sort(key=int)
        trips[V].sort()

    # Output results:
    print("Volts Database Status:")
    print("-"*50)
    print("Directory:",data_path)
    print("Files:",len(files_list),"parquet files found.")
    print("Unique id values: ", ids)
    print("Total number of vehicles: ", len(vehicles))
    print("Total number of complete trips: ", sum(trip_counts.values()))
    print("-"*50)
    print("Trips per vehicle:")
    for V in sorted(trip_counts.keys(),key=int):
        print("     V",V,": ",trip_counts[V], "complete trips")
    print("-"*50)
    
    return files_list, trips_compl


# %%
# Output Database information:
all_files, trip_data = trip_data_info("data/processed")


# %%
vehicle_selection = '14' # type: str
V_files = [f for f in files if ('v_id983V'+vehicle_selection) in f]
V_files.sort()
for f in V_files:
    print(f)


# %%
# Trip Sizes:
data_dir = "data/processed"
cwd = os.getcwd()
data_path = os.path.join(cwd,data_dir)


for f in all_files:
    df = pd.read_parquet(os.path.join(data_path,f))
    print(f, ": ", df.shape)

# Get the shape of the DataFrame (rows, columns)
# %%
