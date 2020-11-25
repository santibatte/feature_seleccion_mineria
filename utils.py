# Funciones auxiliares

#Funcion que calcula el rango intercuartilico: 
def iqr(x):
  return np.subtract(*np.percentile(x, [75, 25]))



# ------------------------------------------------------------------
# Funciones de la seleccion de caracteristicas

# Low variability
def low_variability(df,treshold):
    '''
    treshold = float entre 0 y 1 
    Función que descarta las variables con baja variabilidad intercuartilica, dado un treshold
    recibe un data frame y un treshold 
    devuelve las variables que tienen un rango intercuartilico superior a un porcentaje/treshold del rango intercuartilico global y la lista de variables eliminadas
    '''
    #La media de los rangos intercuartilicos del data frame
    DF=df.copy()
    drop_this=[]
    
    feature_matriz= []
    
    #Faltaria estandarizar las variables. 
    scaler = MinMaxScaler() 
    DF = scaler.fit_transform(DF) 
    DF = pd.DataFrame(data= DF, columns= df.columns )
    
    global_iqr=st.mean(DF.apply(iqr))
    treshold = treshold * global_iqr
    
    for i in range(DF.shape[1]):
        iqr_column = iqr(DF.iloc[:,i])
        if iqr_column < treshold:
            drop_this.append(DF.columns[i])
        
    print('columns dropped: ', drop_this)
    DF.drop(columns= drop_this,inplace=True)
            
            
    return DF
    


# correlation filtering
def correlation_filtering(data_frame, treshold):
    '''
	data_frame es una base de datos de variables numericas
	treshold es el umbral de decision, entre 0 y 1
	
	La funcion tira las variables que esten muy correlacionadas con alguna otra variable de la base.
    '''

    df = pd.DataFrame(data_frame)
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # guarda las columnas que se van a tirar
    to_drop =[]

    for n in range(df.shape[1]): # itera sobre columnas del data frame
        
        if any(upper.iloc[:,n]>treshold):     # si algun elemento de la matriz de correlacion de la columna n es mayor al treshold
            print(upper.columns[n], "sobrepasa el umbral")
            to_drop.append(df.columns[n])     # se agrega para borrarse
            
    print('Columnas borradas:', to_drop)            # columnas que se van a tirar   

    # Drop features 
    df.drop(columns= to_drop,inplace=True)

    return df


# Fast correlation based filtering
def FCB_filtering(X,y,treshold=0.7):
    '''
    X Es un data Frame de features 
    y es una columna de etiquetas 
    treshold es el umbral de decisión, entre 0 y 1
	
	La funcion toma las correlaciones de cada variable del data frame con la columna de etiquetas
	y, para aquellas que esten por encima del umbral, evalua cuales otras variables del data frame
	tienen una correlacion alta con el vector de caracteristicas en cuestion, y las elimina.
    '''


    dfx = pd.DataFrame(X) # base que NO incluye la variable objetivo
    y = pd.DataFrame(y)   # vector de etiquetas

    # 1. Obtener la correlación de cada variable en X con la variable de salida y guardar las que pasen el umbral
    corr_xy = [] # aquí se van a guardar las correlaciones de cada xi con y
    high_corr = [] # aquí se guardan las variables que están por arriba del umbral especificado

    #Lista con las caractrísticas xi más relacionadas con y
    for n in range(dfx.shape[1]): # itera por columnas del data frame
        column = dfx.iloc[:,n]    # guarda la columna
        a = abs(y.corrwith(column))[0] # calcula la correlacion en valor absoluto
        if a > treshold:                     
            high_corr.append(dfx.columns[n])
    max = len(high_corr) # Número de veces que iterará la función

    for iteracion in range(max):


        corr_xy = []

        #Lista de correlaciones de todas las caractrísticas xi con y
        for n in range(dfx.shape[1]): # itera por columnas del data frame
            column = dfx.iloc[:,n]    # guarda la columna
            a = abs(y.corrwith(column))[0] # calcula la correlacion en valor absoluto
            corr_xy.append(a)                

        # 2. Ordenar las variables de mayor a menor correlación con Y
        temp = pd.DataFrame(corr_xy).reset_index()
        temp = temp.rename(columns={0: 'corr'})
        temp = temp.sort_values(by=['corr'], ascending=False)

        #3. Medir la correlación de la mejor variable con las demás
        feature = temp.iloc[0,0] # Característica xi más correlacionado con y
        df_xi = dfx.iloc[:,feature] # Dataframe con el xi más correlacionado con y  
        df_witouth_xi = pd.DataFrame(dfx)
        df_witouth_xi.drop(columns = dfx.columns[feature] ,inplace=True) # Dataframe sin el xi más correlacionado con y

        corr_xi_all = abs(df_witouth_xi.corrwith(df_xi)) # Correlacion de xi con las demás características

        # 4. Tirar variables que estén muy correlacionadas con la primera variable.
        to_drop =[] # vector de variables que se van a borrar

        for n in range(df_witouth_xi.shape[1]): # itera sobre columnas del data frame

                # si algun elemento de la matriz de correlacion de la columna n es mayor al treshold
                if (corr_xi_all.iloc[n]>treshold):  

                    to_drop.append(dfx.columns[n])     # se agrega para borrarse

                else:
                    (corr_xi_all.iloc[n], "no sobrepasa el umbral")

        print('Columnas borradas:', to_drop)            # columnas que se van a tirar   

        # Drop features 
        df_witouth_xi.drop(columns= to_drop,inplace=True)

        # 5. Dataframe de Características
        if iteracion == 0:
            features_df = pd.DataFrame(df_xi)
        elif (iteracion > 0) and (iteracion < max-1):
            features_df = pd.concat([features_df, df_xi], axis=1, sort=False)
        else:
            features_df = pd.concat([features_df, df_witouth_xi], axis=1, sort=False)

        dfx = pd.DataFrame(df_witouth_xi) #Dataframe que se vuelve a evaluar

    return features_df
	


# Forward selection
def forward_selection(X,y,tol=0.001):
    '''
    X Es un data Frame de features 
    y es una columna de etiquetas 
    
    Elije la mejor variable, una a la vez, para un modelo lineal, y va agregando variables. 
    Devuelve las columnas seleccionadas con sus datos, para el train y el test 
    '''
    
    #X= pd.DataFrame(data=X, columns = boston['feature_names']) 
    
    #Y = pd.DataFrame(data=y)
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=15)
    
    ## Convertirlos en panda data frame: 
    
    col_select_train= pd.DataFrame()
    col_select_test = pd.DataFrame()
    max_R_2 = 0.1
    max_R_2_ant = 0
    
    while np.abs(max_R_2 - max_R_2_ant) > tol and X_train.shape[1] > 0:         
        
        if len(col_select_train) == 0:  # Primera vuelta, crea el data frame para la primera variable 
            for i in range (X_train.shape[1]):
                R_cuadrado= []
                X_train_m= X_train.iloc[:,i][:, np.newaxis]
                X_test_m= X_test.iloc[:,i][:, np.newaxis]
                
                modelo=LinearRegression()
                modelo.fit(X_train_m, y_train)
                
                #Voy agregando todos los R_cuadrados
                R_cuadrado.append(modelo.score(X_test_m, y_test) ) 
                
            #Me quedo con el mejor R_Cuadrado: 
            max_R_2_ant = max_R_2
            max_R_2 = max(R_cuadrado)
            max_index = R_cuadrado.index(max_R_2)
                
            #Agrego la columna perteneciente al mejor R cuadrado a mis variables seleccionadas:
            col_select_train= pd.DataFrame(data=X_train.iloc[:,max_index])
            col_select_test= pd.DataFrame(data=X_test.iloc[:,max_index])
            #print('columnas del modelo seleccionado: ', col_select_train.columns)
            
            
            #Eliminar la columna de X_train y X_test
            X_train.drop(X_train.columns[max_index],axis=1,inplace=True)
            X_test.drop(X_test.columns[max_index],axis=1,inplace=True)
            
        else: 
            for i in range (X_train.shape[1]):
                R_cuadrado= []
                
                ## Panda data frame, con column_select_train + la nueva variable a explorar: 
                X_train_m= pd.concat([col_select_train, X_train.iloc[:,i] ], axis=1)
                X_test_m= pd.concat([col_select_test, X_test.iloc[:,i] ], axis=1)

                modelo=LinearRegression()
                modelo.fit(X_train_m, y_train)
                
                #assert sum(X_train_m['CRIM'] == 'Nan') == 0
                
                #print(X_train_m)
                #print(X_train_m.shape)
                #print('estamos en la i:', i)
                #print('coef', modelo.coef_)
                #print('score', modelo.score(X_test_m, y_test))
                
                #Voy agregando todos los R_cuadrados
                R_cuadrado.append(modelo.score(X_test_m, y_test) ) 
                
            #Me quedo con el mejor R_Cuadrado: 
            max_R_2_ant = max_R_2
            max_R_2 = max(R_cuadrado)
            #print('R cuadrado seleccionado es de: ', max_R_2)
            max_index = R_cuadrado.index(max_R_2)

            #Agrego la columna perteneciente al mejor R cuadrado a mis variables seleccionadas:
            col_select_train= pd.concat([col_select_train, X_train.iloc[:,max_index] ], axis=1)
            col_select_test= pd.concat([col_select_test, X_test.iloc[:,max_index] ], axis=1)
            #print('columnas del modelo seleccionado: ', col_select_train.columns)
            
            #col_select_train.append(X_train[:,max_index])
            #col_select_test.append(X_test[:,max_index])

            ## Elimino la columna elegida de X_train y X_test para que no pueda volver a ser elegida
            X_train.drop(X_train.columns[max_index],axis=1,inplace=True)
            X_test.drop(X_test.columns[max_index],axis=1,inplace=True)

    
    print('columnas del modelo seleccionado: ', col_select_train.columns)
    return col_select_train, col_select_test
