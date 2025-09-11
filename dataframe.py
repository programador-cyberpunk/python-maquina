df_vazio = pd.DataFrame()
print("Um bagulho vazio")
print(df_vazio)

print("-" * 30)

lista = ['gracas', 'a', 'satanas','eh','sexta','feira']
df_Lista = pd.DataFrame(lista)
print("\nDataFrame criado a partir de uma lista: ")
print(df_Lista)

print("-" * 30)

dados_dict = {
    'Nome': ['Armando', 'Claudio', 'Alfredo', 'Gabriel', 'Fabricio'],
'Idade': [36, 25, 19, 18, 47],
'Cidade': ['Guarulhos', 'São Paulo', 'Santos', 'Guarulhos', 'São Paulo'],
'Profissão': ['Advogado', 'Técnico', 'Advogado', 'Técnico', 'Advogado']
}

df = pd.DataFrame(dados_dict)
print("\O data frame criado a partir de um dicionario ai:")
print(df)

print("DartaFrame Original: ")
print(df)

print("-" * 35)

serie_nomes = df['Nome']
print("\nSelecionando a coluna 'Nome' (e o resultado é uma Series): ")
print(type(serie_nomes))
print(serie_nomes)

print("-" * 35)
sub_df = df[['Cidade', 'Profissao']]
print("\n Cidade e profissao foram selecionados")
print(type(sub_df))
print(sub_df)

#primeiro metodo pra add colunas novas
df_copia = df.copy()
estados = ['SP','SP','SP','SP','SP']
df_copia['Estado'] = estados
print("Dataframe com a coluna nova 'ESTADO': ")
print(df_copia)

#segundo metodo pra add colunas novas
salarios = [500.00, 4200.00, 6800.00, 3900.00, 9100.00]
df_copia.insert(2,'Salario',salarios)
print('Dataframe com a coluna "Salario" inserida na posicao 2: ')
print(df_copia)

anos_xp = [10,5,2,1,20]
df_novo = df_copia.assign(AnosDeXP=anos_xp)

print("DataFrame original 'df_copia' (nao mudou nada): \n")
print(df_copia)

print("\n Novo dataframe retornado por assign(): \n")
print(df_novo)

df_novo_multi = df_copia.assign(
    AnosDeExperiencia = [10, 5, 2, 1, 20],
    Avaliacao = [8.5, 9.0, 7.8, 8.1, 9.5]
)
print("\n Adicionando multpilas colunas com assign(): ")
print(df_novo_multi)

#usando dataframe.loc
bonus_valores = [1500,800,1300,750,1800]
df_copia.loc[:, 'Bonus'] = bonus_valores

print("\nDatafram com a nova coluna 'Bonus' add via .loc: \n")
print(df_copia)
      
print("DataFrame antes de remover a coluna 'Bonus':")
print(df_copia)
# Removendo a coluna 'Bonus'. A operação é feita "in-place", modificando␣df_copia.

df_copia.drop(columns=['Bonus'], inplace=True)
print("\nDataFrame após remover a coluna 'Bonus':")
print(df_copia)