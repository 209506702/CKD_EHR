# CKD_EHR

Electronic Health Records (EHR)-based disease prediction models have demon- strated significant clinical value in promoting precision medicine and enabling early intervention.

However, existing large language models face two major challenges: insufficient representation of medical knowledge and low efficiency in clinical deployment. 

To address these challenges, this study proposes the CKD-EHR (Clinical Knowledge Distillation for EHR) framework, which achieves efficient and accurate disease risk prediction through knowledge distil- lation techniques. Specifically, this method first performs domain adaptation fine-tuning of Qwen2.5-7B using medical knowledge-enhanced data. Subse- quently, interpretable soft labels are generated through a multi-granularity attention distillation mechanism. Finally, the distilled knowledge is trans- ferred to a lightweight BERT model. 

This innovative solution not only greatly improves resource utilization efficiency but also significantly enhances the accuracy and timeliness of diagnosis, providing a practical technical approach for resource optimization in clinical settings.

Frame

![fig1](https://github.com/user-attachments/assets/7e8e3d6c-e687-477b-b1ff-dd1d99361634)
