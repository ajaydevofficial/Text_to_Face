from django.db import models


# Create your saved_models here.
class Image(models.Model):
    image = models.ImageField(upload_to="dataset")
    gender = models.CharField(max_length=100, blank=True, default='')
    race = models.CharField(max_length=100, blank=True, default='')
    age = models.CharField(max_length=100, blank=True, default='')
    hair = models.CharField(max_length=100, blank=True, default='')
    physic = models.CharField(max_length=100, blank=True, default='')

    class Meta:
        db_table = 'Image'
