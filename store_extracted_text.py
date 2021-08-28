from collections import OrderedDict
import pandas as pd

def store_text(extracted_text, image_path, storage_dict):
    storage_dict = OrderedDict() if len(storage_dict) == 0 else storage_dict.to_dict()
    counter = 0
    txt_og = extracted_text
    txt_clean = txt_og.replace(" ", "")
    txt_clean = txt_clean.split('\n')
    for field in txt_clean:
            field_split = field.split(":")
            if field_split[0] == "":
                continue
            if len(field_split) == 1 and counter == 0:
                if "Title" in storage_dict:
                    storage_dict["Title"].append(field_split[0])
                else:
                    storage_dict["Title"] = [field_split[0]]
            if len(field_split) == 1 and counter > 0:
                if "End" in storage_dict:
                    storage_dict["End"].append(field_split[0])
                else:
                    storage_dict["End"] = [field_split[0]]

            if len(field_split) == 2:
                if field_split[0] in storage_dict:
                    storage_dict[field_split[0]].append(field_split[1])
                else:
                    storage_dict[field_split[0]] = [field_split[1]]
            if len(field_split) > 2:
                raise Exception("More than two fields given, skipping")
            counter += 1
    if "ImagePath" in storage_dict:
        storage_dict['ImagePath'].append(image_path)
    else:
        storage_dict['ImagePath'] = [image_path]

    output_df = pd.DataFrame(storage_dict)
    output_df = output_df.set_index("DocNo")

    return output_df, storage_dict


