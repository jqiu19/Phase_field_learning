#python free_energy_1.py --epochs 5000 --features 100 --layers 20 --ntrain 50000 --network "mlp" --seed 0 --device 7
#python free_energy_1.py --epochs 5000 --features 100 --layers 10 --ntrain 50000 --network "modifiedmlp" --seed 0 --device 6
#python free_energy_1.py --epochs 5000 --kanshape 16 --degree 100 --ntrain 50000 --network "sinckan" --len_h 6 --init_h 2.0 --decay 'inverse' --skip 0 --seed 0 --device 5

for j in 3 5 10 50 100 200 300 400 500 600
do
    python free_energy_1.py --epochs 5000 --kanshape 2 --degree $j --ntrain 500000 --network "sinckan" --len_h 3 --init_h 0.5 --decay 'inverse' --skip 0 --seed 0 --device 5
done

for i in 20 50 100 200 300 500 700
do
    python free_energy_1.py --epochs 5000 --features $i --layers 10 --ntrain 500000 --network "modifiedmlp" --seed 0 --device 6
done

