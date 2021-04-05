
def write_to_file(filename, num_points_deviation_dicts, function_labels):
    with open(filename, 'w') as file:
        file.write("num points    deviation in %\n")
        for i in range(len(function_labels)):
            file.write("\n"+function_labels[i]+"\n")
            for num_points, deviation in num_points_deviation_dicts[i].items():
                file.write(str(num_points)+"    "+str(deviation)+"\n")

