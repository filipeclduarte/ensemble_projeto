import requests
import pickle

def main():
    dias_mes_2014_2015 = [
        31, 28, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31,
    ]

    dias_mes_2016 = [
        31, 29, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31,
    ]

    meses = [
        '01', '02', '03', '04', '05', '06', 
        '07', '08', '09', '10', '11', '12',
        ]

    anos = [2014, 2015, 2016]

    anos_dict = dict.fromkeys(anos)
    for ano in anos:   
        print(f'Ano: {ano}')
        if ano == 2016:
            meses_dict = dict.fromkeys(meses)
            for mes in meses:
                print(f'Mês: {mes}')
                dias_temp = [f'0{i}' for i in range(1,10)] + [f'{i}' for i in range(10, dias_mes_2016[meses.index(mes)]+1)]  
                print('dias_temp: ', dias_temp)
                dias_dict = dict.fromkeys(dias_temp)
                for dia in dias_dict.keys():
                    print('dia: ', dia)
                    url_temp = f'https://apidatos.ree.es/en/datos/demanda/evolucion?start_date={ano}-{mes}-{dia}T00:00&end_date={ano}-{mes}-{dia}T23:59&time_trunc=hour&geo_trunc=electric_system&geo_limit=peninsular&geo_ids=8741'
                    response = requests.get(url_temp)
                    dados_salvar = response.json()
                    dias_dict[dia] = dados_salvar

                meses_dict[mes] = dias_dict
            anos_dict[ano] = meses_dict

        else:
            meses_dict = dict.fromkeys(meses)
            print(meses_dict)
            for mes in meses:
                print(f'Mês: {mes}')
                dias_temp = [f'0{i}' for i in range(1, 10)] + [f'{i}' for i in range(10, dias_mes_2014_2015[meses.index(mes)]+1)]  
                print('dias_temp: ', dias_temp)
                dias_dict = dict.fromkeys(dias_temp)
                for dia in dias_dict.keys():
                    print('dia: ', dia)
                    url_temp = f'https://apidatos.ree.es/en/datos/demanda/evolucion?start_date={ano}-{mes}-{dia}T00:00&end_date={ano}-{mes}-{dia}T23:59&time_trunc=hour&geo_trunc=electric_system&geo_limit=peninsular&geo_ids=8741'
                    response = requests.get(url_temp)
                    dados_salvar = response.json()
                    dias_dict[dia] = dados_salvar
                
                meses_dict[mes] = dias_dict
            anos_dict[ano] = meses_dict
                   
        with open(f'dados/{ano}.pkl', 'wb') as file:
                        pickle.dump(anos_dict[ano], file)


if __name__ == '__main__':
    main()

# dados são armazenados da seguinte forma:
# dados['ano']['mes']['dia']['included'][0]['attributes']['values'][:24]