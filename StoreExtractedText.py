import os
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from main import PROJECT_DIRECTORY


class PostProcessAndStore:
    def __init__(self, extracted_text, config, image_path):
        self.extracted_text = extracted_text
        self.config = config
        self.image_path = image_path
        self.counter = 0

    def prepare_storage_dataframe(self):
        """
        Prepare storage dataframe
        :return:
        """
        storage_dataframe_path = self.config.get('storage_dataframe_path')
        if storage_dataframe_path is None:
            return OrderedDict()
        storage_dataframe_path = os.path.join(PROJECT_DIRECTORY, storage_dataframe_path)
        if not os.path.exists(storage_dataframe_path):
            return OrderedDict()
        storage_dataframe = pd.read_csv(storage_dataframe_path)
        storage_dict = storage_dataframe.to_dict("list")
        return storage_dict

    def post_process(self):
        """
        Post process text, clean up extracted text
        :param extracted_text: [str]
        :return: txt_clean [List]
        """
        txt_og = self.extracted_text
        txt_clean = txt_og.replace(" ", "")
        txt_clean = txt_clean.split('\n')

        return txt_clean

    def create_storage_dict(self, txt_clean, counter, storage_dict):
        """
        Create storage dict for extracted text
        :param txt_clean:
        :param counter:
        :param storage_dict:
        :return:
        """
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
                print("More than two fields given, skipping")
            counter += 1
        if "ImagePath" in storage_dict:
            storage_dict['ImagePath'].append(self.image_path)
        else:
            storage_dict['ImagePath'] = [self.image_path]

        return storage_dict

    def store_text(self):
        """
        Run for preparing storage dataframe, text postprocess and structured storage
        :return:
        """
        storage_dict = self.prepare_storage_dataframe()
        # Clean Text
        txt_clean = self.post_process()
        # Store extracted text in structured object, dict.
        storage_dict = self.create_storage_dict(txt_clean=txt_clean, counter=self.counter, storage_dict=storage_dict)

        output_df = pd.DataFrame(storage_dict)

        return output_df, storage_dict


