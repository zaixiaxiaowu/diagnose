from django.db import models


class Image1(models.Model):
    photo = models.ImageField(null=True, blank=True)

    def __str__(self):
        return self.photo.name
