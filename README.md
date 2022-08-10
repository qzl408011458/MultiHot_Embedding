# MultiHot_Embedding
This is a code repository of the submitted AAAI 2023 paper: *MultiHot Embedding: A Multiple Activation Embedding Model for Continuous Features in Deep Learning*.

Please cite our paper if you use the code in your work. The suggested citation is 'P. Zhang, Z. Ma, Z. Qing, Y. Zhao (2022). MultiHot Embedding: A Multiple Activation Embedding Model for Continuous Features in Deep Learning. arxiv xxx'  

For task1 and task3, you can use the following commands to reproduce our
 best models.

```python
python train_task1.py --module efde --bins 200 --intv 5 --emb_siz 200 --scalar standard
```

```python
python train_task3.py --module ewde --bins 200 --intv 5 --emb_siz 100 --scalar standard --hid_size 128
```

For task2, the processed model data is provided since the original dataset can not be open to the public for the industrial privacy protocol. 
The running command for our best model in task2 is as follows:

```python
python train_task2.py --module efde --bins 510 --intv 16 --emb_siz 200 --scalar standard --hid_size 300
```






