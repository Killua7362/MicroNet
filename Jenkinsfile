pipeline {
    environment {
        PROJECT = "clowder-403113"
        APP_NAME = "test-app"
        REPO_NAME = "testing"
        REPO_LOCATION = "asia-south1"
        IMAGE_NAME = "${REPO_LOCATION}-docker.pkg.dev/${PROJECT}/${REPO_NAME}/${APP_NAME}"
        CRED_ID = 'gcloud-creds'
        REGION = 'asia-south1'
    }
    agent any
    stages {
        stage('Pull Git') {
            steps {
                echo 'Cloning..'
                deleteDir() // Clean the workspace
                checkout scm
}
        }
        stage('Build and push docker image'){
                steps{
                    script{
                    sh 'docker build -t ${IMAGE_NAME}:lastest -f Dockerfile .'
                    withCredentials([file(credentialsId: "${CRED_ID}", variable: 'GCR_CRED')]){
                        sh 'cat "${GCR_CRED}" | docker login -u _json_key --password-stdin https://"${REPO_LOCATION}"-docker.pkg.dev'
                        sh 'docker push ${IMAGE_NAME}:lastest'
                        sh 'docker logout https://${REPO_LOCATION}-docker.pkg.dev'
                    }    
                    try{
                        sh 'docker rmi ${IMAGE_NAME}:lastest'
                    }catch(Exception e){
                        echo "oops"
                    }     
                    }
                }
            }
        stage('Create gke cluster'){
            steps{
                script{
                        withCredentials([file(credentialsId: "${CRED_ID}", variable: 'GCR_CRED')]){
                            sh("gcloud auth activate-service-account --key-file=${GCR_CRED}")
                            try{
                                sh "gcloud container clusters create-auto ${APP_NAME} --region ${REGION} --service-account=clowder@clowder-403113.iam.gserviceaccount.com"
                            }catch (Exception e){
                                echo "We got error"
                            }
                            sh "gcloud container clusters get-credentials ${APP_NAME} --region ${REGION}"
                            sh "kubectl apply -f kubernetes/backend.yaml"
                            sh "kubectl apply -f kubernetes/service.yaml"
                            def out = sh(returnStdout:true, script: 'kubectl get services')
                            echo out
                        }
                }
            }
        }
  
        }
    }