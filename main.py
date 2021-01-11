import numpy as np
import math
import matplotlib.pyplot as plt



class Kohonen:
    def __init__(self, learning_rate, number_neuron,number_epoch,sigma0,time_constant1,time_constant2):
        self.sigma0 = sigma0
        self.time_constant1 = time_constant1
        self.time_constant2 = time_constant2
        self.number_epoch = number_epoch
        if(number_neuron%3==1): ## Parametre olarak verilen nöron sayısı tek sayı ise, nöronları 2 boyutlu eksende konumlandırmak için
            #1 ekleyerek çift sayı yaparız.
            number_neuron+2
        if (number_neuron % 3 == 2):
            number_neuron+1
        self.w3 = np.zeros(shape=(3,1))
        self.number_neuron = number_neuron  ##Parametre olarak main fonksiyonda verilen nöron sayısını classımızın içine atıyoruz.

        self.learning_rate = learning_rate  ##Parametre olarak main fonksiyonda verilen öğrenme katsayısını classımızın içine atıyoruz.

        ##We will create 81 neurons which means we're going to need 81 weights

        self.w1 = [np.random.normal(0, 0.1, size=(3, 1)) for i in range(round(
            self.number_neuron ))]  # Created 27 amount of 3 dimensional weights that is distributed by gaussian normal distribution that is mean point is 4 and standart deviation is 0.5



        self.w = self.w1  # üç farklı ağırlık listesini tek bir ağırlık listesinde birleştiririz.

        self.w =np.array(self.w)
        self.x_size = math.sqrt(number_neuron) #nöron sayısının karekökü, nöron düzlemimizde bir kenarda kaç nöron olduğunu belirleyecek.
        self.x_size = math.ceil(self.x_size) #Eğer nöron sayımız 64 ise nöronlarımız 8x8 bir matrise göre konum alır. Eğer 65 nöronumuz varsa 9x9 bir matrise göre konum alır. Geri kalan nöronlara 0 koyarız.

        self.initial_w = self.w
        self.initial_w = np.array(self.initial_w)
        self.neuron_matrix = np.zeros(shape=(self.number_neuron,self.number_neuron))

        self.neuron_konum_düzlemi = np.zeros(shape=(self.x_size,self.x_size))#64 nöronumuz varsa, nöronların konumlarını 8x8 kare matrise yerleştirdiğimizde nöronlar arası mesafeyi hesaplamak için oluturduk.

        for j in range(self.number_neuron):
            for i in range(self.x_size):
                for k in range(self.x_size):
                    self.neuron_konum_düzlemi[i][k] = math.sqrt((int(j/self.x_size)-i)*(int(j/self.x_size)-i)+((j%self.x_size)-k)*((j%self.x_size)-k)) #Alt satırda açıklaması mevcut

            self.neuron_matrix[j] = self.neuron_konum_düzlemi.flatten() # Üstteki üç for yapısının içindeki işlemler ile neuron_matrix, hocanın derste gösterdiği gibi, nöronların diğer nöronlara olan uzaklığını tutuyor.
            #neuron_matrix[0][4]=4 çünkü 0. nöron ile 4. nöron arasındaki uzaklık 4 birim. neuron_matrix[0][7] 0. nöron ile 7. nöron arasındaki uzaklığı gösterir. Eğer nöron sayımız 25 ise, nöronlarımızı 5x5 kare matrise konumlandırırız.
            #böylece 0. nöron 0-0 koordinatlarında olur, 7. nöron (1-2) koordinatlarında olur. Böylece neuron_matrix[0][7] = 2.23606797749979( sqrt(2^2+1^2)) değeri olur. neuron_matrix[2][14] = 2.82 çünkü (0,2) koordinatındaki
            # ve (2,4) koordinatındaki nöronların arasındaki mesafe 2.82 birimdir. Böylelikle main fonksiyonda nöron sayısı parametresini değiştirmemize karşılık her zaman çalışan bir uzaklık bulma algoritmamız olur.
            #neuron_matrix[0] koordinatı (0,0) olan ilk nöronun diğer tüm nöronlara olan uzaklığını depolayan numpy array değişkenidir. neuron_matrix[1] ise (0,1) koordinatındaki ikinci nöronun diğer tüm nöronlara olan uzaklıklarını depolar.



    def train(self,x,class_list):

        self.d = np.zeros(shape=(self.number_neuron))

        self.neuron_class_list = np.zeros(shape=(self.number_neuron))
        self.w_transpose = np.zeros(shape=(1,3))
        self.h = np.zeros(shape=(self.number_neuron))

        self.class_list = np.array(class_list)





        for i in range(self.number_epoch):
            self.sigma_k = self.sigma0*np.exp(-i/self.time_constant1)   #sigma(k) değerini buluruz time_constant1 kitaptaki tau1 değeridir.
            self.learning_rate_k = self.learning_rate * np.exp(-i / self.time_constant2)  # İterasyon sayısına bağlı değişen öğrenme oranı. 0.01'den az olması istenmez.

            for k in range(len(x)):
                for j in range(self.number_neuron): #Bu for döngüsüyle yarışma yaparız. Kazanan nöronu bulmaya çalışırız. Her nöronun giriş ile iç çarpımını buluruz.
                    self.w_transpose = np.transpose(self.w[j])
                    self.euclidean_norm = (x[k][0]-self.w[j][0])**2+(x[k][1]-self.w[j][1])**2+(x[k][2]-self.w[j][2])        #Kazanan nöronu buluruz. Öklid normunu bulur, en düşük sonuç kazanan nörona aittir.
                    self.d[j] = self.euclidean_norm

                self.index_max =np.argmin(self.d)  # En düşük sonucun indexi, index_max'e atanır.w[index_max] nöronu kazanmıştır.
                                                     #  index_map nöronunu ve komşu nöronları değiştiricez. Değiştirmemiz için bu nöronun diğer her nörona uzaklığını bulmamız gerek.

                self.neuron_class_list[self.index_max] = self.class_list[k]  #Eğittiğimiz verimizin sınıfı ne ise, kazanan nörona, verinin sınıfını atarız. Mesela 1. verimizin sınıfı 3 olsun. 1. verimizi 15. nöron kazandıysa
                #15. nöron 3. sınıfı temsil eder. Bunları numpy arrayi halinde depolarız.


                
                for z in range(self.number_neuron):     #Kazanan nöronumuz belli. Şimdi nöronun komşu nöronlarına olan etkisini bulup ağırlık güncelleyeceğiz.
                    #  Nöronların, index_map nöronuna uzaklıklarını bulmamız lazım.

                    self.h[z] = np.exp(-((self.neuron_matrix[self.index_max][z])**2)/(2*(self.sigma_k**2))) #self.neuron_matrix[self.index_max][z] kazanan nöron ile her nöron arasındaki mesafeleri returnler.
                    #  Kazanan nöronun diğer nöronlara olan mesafelerine göre değiştiren fonksiyona alıyoruz


                    self.w[z] = self.w[z]+self.learning_rate_k*self.h[z]*(x[k]-self.w[z]) #Ağırlık güncelliyoruz.

            



    def test(self,test_list): #  Test kümesini parametre olarak veririz. Test kümemiz de 3 boyutlu bir numpy arraydir. Sınıflandırmayı doğru yapabilecek mi kontrol edeceğiz.
        self.dot_product = np.zeros(shape=(self.number_neuron)) #  Bir veri için bütün nöronlarla iç çarpımlarını saklar, en büyük iç çarpımı bu array listesinden buluruz.
        self.equality = 0
        self.acc_list = np.zeros(shape=(self.number_neuron))
        for k in range(len(test_list)): #  Bu for her veri kümesini test etmek için
            for j in range(self.number_neuron):  # Bu for döngüsüyle yarışma yaparız. Kazanan nöronu bulmaya çalışırız. Her nöronun, test verisi ile iç çarpımını buluruz.

                self.w_transpose = np.transpose(self.w[j])

                self.euclidean_norm = (test_list[k][0] - self.w[j][0]) ** 2 + (test_list[k][1] - self.w[j][1]) ** 2 + (
                            test_list[k][2] - self.w[j][2])**2  #  nöron ile girişin öklid normunu alıyoruz.

                self.dot_product[j] = self.euclidean_norm #  Bulduğumuz öklid normunu bir listeye atıyoruz.

            self.index_min_test = np.argmin(self.dot_product)   #'''  En düşük öklid normunu buluruz, indexini index_min_test' eşitleriz. Verdiğimiz giriş için girişe en yakın nöronun indexini bulmuş oluruz.'''
            print("Olması gereken sınıf:"+str(self.class_list[k])+", "+str(test_list[k])+" verisinin ağın tespit ettiği sınıfı: "+str(self.neuron_class_list[self.index_min_test])+"\n\n\n") #'''test verisinin hangi sınıfa ait olduğunu yazdırıyoruz.'''

            if (class_list[k]==self.neuron_class_list[self.index_min_test]):
                self.equality+=1
        print("Accuracy: "+str(self.equality/len(test_list)))

train_list1 = [np.random.normal(0,0.1,size=(3,1)) for i in range(50)]     #1. sınıftan eğitim kümesi merkezi 0, standart sapması 0.1 olan ve gausian dağılımıyla 150 adet 3 boyutlu için nokta üretiriz.'''
train_list2 = [np.random.normal(1,0.1,size=(3,1)) for i in range(50)]      #2. sınıftan eğitim kümesi
train_list3 = [np.random.normal(-1,0.1,size=(3,1)) for i in range(50)]         #3. sınıftan eğitim kümesi
train_list4 = [np.random.normal(0,0.1,size=(3,1)) for i in range(50)]     #1. sınıftan eğitim kümesi ''
train_list5 = [np.random.normal(1,0.1,size=(3,1)) for i in range(50)]      #2. sınıftan eğitim kümesi
train_list6 = [np.random.normal(-1,0.1,size=(3,1)) for i in range(50)]      #3. sınıftan eğitim kümesi
train_list7 = [np.random.normal(0,0.1,size=(3,1)) for i in range(50)]     #1. sınıftan eğitim kümesi
train_list8 = [np.random.normal(1,0.1,size=(3,1)) for i in range(50)]      #2. sınıftan eğitim kümesi
train_list9 = [np.random.normal(-1,0.1,size=(3,1)) for i in range(50)]      #3. sınıftan eğitim kümesi

train_list =  train_list1+train_list2+train_list3+train_list4+train_list5+train_list6+train_list7+train_list9+train_list8   #üç farklı listeyi tek bir listede birleştiririz.'''
train_list = np.array(train_list)


class_list1 = [np.array([1]) for i in range(50)] #Eğitim kümesinin sınıf bilgilerini oluşturuyoruz. İlk 50 eğitim verisinin sınıfı 1, 50-100. veriler 2. sınıfa ait, 100-150. veriler 3. sınıfa ait.
class_list2 = [np.array([2]) for i in range(50)]
class_list3 = [np.array([3]) for i in range(50)]
class_list4 = [np.array([1]) for i in range(50)] #bunu 3 kere tekrarlarız böylece karma bir şekilde eğitime sunulmuş olsun.
class_list5 = [np.array([2]) for i in range(50)]
class_list6 = [np.array([3]) for i in range(50)]
class_list7 = [np.array([1]) for i in range(50)]
class_list8 = [np.array([2]) for i in range(50)]
class_list9 = [np.array([3]) for i in range(50)]
class_list = class_list1+class_list2+class_list3+class_list4+class_list5+class_list6+class_list7+class_list9+class_list8 #Eğitim kümesi verilerinin sınıf bilgilerini tek bir numpy array'de birleştirdik.
class_list = np.array(class_list)



a = Kohonen(learning_rate= 0.1,number_neuron=25,number_epoch=40,sigma0 = 2,time_constant1 = 20,time_constant2 = 20) #Ağımızı oluşturuyoruz. Parametreleri buradan değiştirebiliriz.
a.train(train_list,class_list) #ağımızı train_list ile eğitiriz, class_list, eğitim verileinin sınıf bilgilerini içerir.

test_list1 = [np.random.normal(0,0.1,size=(3,1)) for i in range(50)]##Test kümesi için merkezi 0, standart sapması 0.1 olan ve gausian dağılımıyla 50 adet 3 boyutlu nokta üretiriz.
test_list2 = [np.random.normal(1,0.1,size=(3,1)) for i in range(50)]##Test kümesi için merkezi 1, standart sapması 0.1 olan ve gausian dağılımıyla 50 adet 3 boyutlu nokta üretiriz.
test_list3 = [np.random.normal(-1,0.1,size=(3,1)) for i in range(50)]##Test kümesi için merkezi -1, standart sapması 0.1 olan ve gausian dağılımıyla 50 adet 3 boyutlu nokta üretiriz.
test_list = test_list1+test_list2+test_list3
test_list = np.array(test_list)

a.test(test_list) #Test kümesini test ederiz. Test kümemizde 50 adet 1. sınıfa ait veri, 50 adet 2. sınıfa ait veri, 50 adet 3. sınıfa ait veri mevcut.

fig=plt.figure()
ax= fig.add_subplot(111, projection='3d')
ax.scatter(a.w[:,0], a.w[:,1], a.w[:,2], c='r', marker='o')
ax.scatter(a.initial_w[:,0], a.initial_w[:,1], a.initial_w[:,2], c='b', marker='^')
ax.scatter(test_list[:,0],test_list[:,1],test_list[:,2], c='g')
#ax.scatter(test_list[:,0], test_list[:,1], test_list[:,2], c='r', marker='o')
plt.title("Green dots are test datas, blue dots are initial weights, red dots are final weights..")
plt.show()
