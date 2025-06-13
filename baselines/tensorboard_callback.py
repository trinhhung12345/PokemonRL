import os
import json

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from einops import rearrange, reduce

def merge_dicts(dicts):
    sum_dict = {}
    count_dict = {}
    distrib_dict = {}

    for d in dicts:
        for k, v in d.items():
            if isinstance(v, (int, float)): 
                sum_dict[k] = sum_dict.get(k, 0) + v
                count_dict[k] = count_dict.get(k, 0) + 1
                distrib_dict.setdefault(k, []).append(v)

    mean_dict = {}
    for k in sum_dict:
        mean_dict[k] = sum_dict[k] / count_dict[k]
        distrib_dict[k] = np.array(distrib_dict[k])

    return mean_dict, distrib_dict

class TensorboardCallback(BaseCallback):

    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None

    def _on_training_start(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, 'histogram'))

    def _on_step(self) -> bool:
        
        # Lấy chỉ số của một môi trường để kiểm tra (ví dụ: môi trường đầu tiên)
        env_idx_to_check = 0
        
        if self.training_env.get_attr("step_count", indices=[env_idx_to_check])[0] % self.training_env.get_attr("max_steps", indices=[env_idx_to_check])[0] == 0:
            all_infos = self.training_env.get_attr("agent_stats")
            if all_infos and all_infos[0]: # Đảm bảo agent_stats không rỗng
                all_final_infos = [stats[-1] for stats in all_infos if stats] # Chỉ lấy nếu stats không rỗng
                if all_final_infos:
                    mean_infos, distributions = merge_dicts(all_final_infos)
                    
                    for key, val in mean_infos.items():
                        self.logger.record(f"env_stats/{key}", val)

                    for key, distrib in distributions.items():
                        self.writer.add_histogram(f"env_stats_distribs/{key}", distrib, self.n_calls)
                        self.logger.record(f"env_stats_max/{key}", max(distrib))

            # --- SỬA LỖI explore_map ---
            # Gọi phương thức `create_exploration_memory` để lấy hình ảnh trực quan
            # Phương thức trả về một danh sách, ta lấy phần tử đầu tiên [0]
            explore_map_image = self.training_env.env_method("create_exploration_memory")[0]
            
            # Ghi log hình ảnh trực tiếp. Dữ liệu có dạng (H, W, C), cần chuyển thành (C, H, W) cho TensorBoard
            self.logger.record(
                "trajectory/exploration_memory", 
                Image(explore_map_image.transpose(2, 0, 1), "CHW"), 
                exclude=("stdout", "log", "json", "csv")
            )

            # --- VÔ HIỆU HÓA PHẦN GÂY LỖI TIẾP THEO ---
            # Môi trường không có thuộc tính "current_event_flags_set", nên chúng ta tạm thời vô hiệu hóa phần này
            # list_of_flag_dicts = self.training_env.get_attr("current_event_flags_set")
            # merged_flags = {k: v for d in list_of_flag_dicts for k, v in d.items()}
            # self.logger.record("trajectory/all_flags", json.dumps(merged_flags))

        return True
    
    def _on_training_end(self):
        if self.writer:
            self.writer.close()

