# SAR

Bu repository, Sentetik Açıklıklı Radar (SAR) görüntülerinde hedef tespiti amacıyla YOLOv5 tabanlı derin öğrenme modellerinin uygulanmasını kapsamaktadır. Çalışma kapsamında iki yaygın SAR veri kümesi kullanılmıştır: SSDD (SAR Ship Detection Dataset) gemi tespiti için ve SAR-AIRcraft-1.0 uçak tespiti için. Tüm deneyler Google Colab ortamında gerçekleştirilmiş olup eğitim ve test süreçleri YOLOv5 mimarisi kullanılarak yürütülmüştür. Ayrıca, modelin karar verme süreçlerini görselleştirmek ve yorumlanabilirliği artırmak amacıyla Class Activation Mapping (CAM) yöntemleri uygulanmıştır.


# SSDD – SAR Ship Detection Dataset

SSDD veri kümesi, SAR görüntülerinde gemi tespiti için yaygın olarak kullanılan bir referans veri setidir. RadarSat-2, TerraSAR-X ve Sentinel-1 gibi farklı SAR sensörlerinden elde edilen görüntüler içermektedir. Veri kümesi toplamda 1,160 görüntü ve 2,456 gemi hedefi içermektedir
Veri Kümesi İndirme Bağlantısı: https://github.com/TianwenZhang0825/Official-SSDD
T. Zhang et al., "SAR Ship Detection Dataset (SSDD): Official Release and Comprehensive Data Analysis," Remote Sens., vol. 13, no. 18, pp. 1–41, 2021, Art. no. 3690.


## SSDD – Tespit Sonuçları

<img width="856" height="531" alt="image" src="https://github.com/user-attachments/assets/e83492a6-ca7b-48e9-ba7f-f281a3e84e7d" />








# SAR-AIRcraft-1.0 Veri Kümesi

SAR-AIRcraft-1.0 veri kümesi, Çin'in Gaofen-3 (GF-3) SAR uydusundan elde edilen yüksek çözünürlüklü görüntülerde uçak tespiti ve sınıflandırması için oluşturulmuştur. Veri seti 4,368 görüntü ve toplam 16,463 uçak örneği içermektedir. Görüntü boyutları 800×800 piksel ile 1500×1500 piksel arasında değişmekte olup 7 farklı uçak sınıfı bulunmaktadır: A220, A320/321, A330, ARJ21, Boeing737, Boeing787 ve Other. Bu veri kümesi, SAR görüntülerinde çok sınıflı hedef tespiti için geliştirilmiş en kapsamlı açık kaynak veri kümelerinden biridir.
Veri Kümesi İndirme Bağlantıları: https://radars.ac.cn/web/data/getData?newsColumnId=f896637b-af23-4209-8bcc-9320fceaba19 (Resmi Kaynak - RADARS) 

WANG Zhirui, KANG Yuzhuo, ZENG Xuan, et al. SAR-AIRcraft-1.0: High-resolution SAR aircraft detection and recognition dataset[J]. Journal of Radars, 2023, 12(4): 906–922. doi: 10.12000/JR23043

(indirmede sorun yaşarsanız sorunlarınızı söyleyebilirsiniz) cakiresra140@gmail.com


## SAR-AIRcraft-1.0 – Tespit Sonuçları

<img width="751" height="751" alt="image" src="https://github.com/user-attachments/assets/863c1a8f-d8d8-4817-8efd-e00a022acad2" />






# CAM
Modelin karar verme süreçlerini anlamak ve hangi bölgelere odaklandığını görselleştirmek amacıyla Class Activation Mapping (CAM) yöntemleri uygulanmıştır. Bu çalışmada pytorch-grad-cam kütüphanesi kullanılarak Grad-CAM, Grad-CAM++, Score-CAM ve Eigen-CAM gibi farklı CAM teknikleri YOLOv5 modeline adapte edilmiştir. Bu yöntemler sayesinde modelin SAR görüntülerinde hedef tespiti yaparken hangi özelliklere ağırlık verdiği ısı haritaları (heatmap) ile görselleştirilebilmektedir.

CAM yöntemlerini uygulamada aşağıdaki Pytorch-CAM githup deposundan yola çıkarak kodların temeli oluşturulmuştur.
Pytorch-CAM Kütüphanesi: https://github.com/jacobgil/pytorch-grad-cam


# Eğitim Ortamı
Tüm eğitim süreçleri Google Colab platformunda gerçekleştirilmiştir. Eğitimler sırasında NVIDIA A100 GPU kullanılmış, PyTorch framework'ü üzerinden Python programlama dili ile kodlanmıştır. Eğitim sırasında mosaic augmentation, copy-paste augmentation, erken durdurma (early stopping) ve özelleştirilmiş hiperparametreler kullanılarak modelin SAR görüntülerine özgü özelliklerini öğrenmesi sağlanmıştır. Tüm deneyler tekrarlanabilir olacak şekilde yapılandırılmış ve sonuçlar detaylı metriklerle raporlanmıştır.


# YOLOv5 deposunu klonlayın
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt 
!!! eğitim için YOLOv5 deposunu klonlayn
