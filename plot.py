import matplotlib.pyplot as plt

x = [x/10 for x in range(-10, 11, 1)]
y = [-0.061327466,-0.062050762,-0.062607679,-0.062960819,-0.0630732,-0.062910433,-0.062443178,-0.061649691,-0.060518163,-0.05904861,-0.057254055,-0.055160879,-0.052808306,-0.050247123,-0.047537784,-0.044748056,-0.041950307,-0.03921842,-0.036624197,-0.034233108,-0.032099371
]
num = 10
plt.plot(x, y)
plt.xlabel("Lambda")
plt.ylabel('Correlation coefficient')
plt.title('delicious CF N='+str(num))
plt.show()
