from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Örnek veri seti oluşturma
data = {'Şarkı Adı': ['Şarkı1', 'Şarkı2', 'Şarkı3', 'Şarkı4', 'Şarkı5'],
        'Tür': [1, 2, 1, 2, 3],
        'Tempo': [120, 130, 125, 135, 128]}

df = pd.DataFrame(data)

# k-NN modelini eğitme
X = df[['Tür', 'Tempo']]
knn = NearestNeighbors(n_neighbors=2)
knn.fit(X)

# Örnek bir şarkı seçme (örneğin ilk şarkı)
test_data = [1, 120]

# En yakın komşuları bulma
distances, indices = knn.kneighbors([test_data])
recommended_songs = df.iloc[indices[0]]

print("Önerilen Şarkılar:")
print(recommended_songs['Şarkı Adı'])
